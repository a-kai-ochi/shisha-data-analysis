#!/usr/bin/env python3
"""Scraper utilities for Shisha Cafe & Bar LAGOS flavor-review articles."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
import re
import tempfile
import time
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import pandas as pd
import requests
from bs4 import BeautifulSoup, Tag

SOURCE_SITE = "shisha_lagos"
SOURCE_TYPE = "editorial_review"
CATEGORY_LABEL = "フレーバーレビュー"
DEFAULT_START_URL = "https://shisha-lagos.com/category/%E3%83%95%E3%83%AC%E3%83%BC%E3%83%90%E3%83%BC%E3%83%AC%E3%83%93%E3%83%A5%E3%83%BC/"
DEFAULT_ALLOWED_DOMAIN = "shisha-lagos.com"
SCRAPER_VERSION = "0.2.0"

LIST_CONTAINER_SELECTOR = "div.blog_list"
LIST_ITEM_SELECTOR = "article.item"
LIST_LINK_SELECTOR = "h3.title a.title_link"
LIST_DATE_SELECTOR = "p.meta time.entry-date.updated"
LIST_DESC_SELECTOR = "p.desc span"
NEXT_PAGE_SELECTOR = "div.page_navi a.next.page-numbers"

ARTICLE_CONTAINER_SELECTORS = [
    "div.post_content.clearfix",
    "div.post_content",
]
ARTICLE_ROOT_SELECTORS = [
    "#article",
    "article",
    "main",
]
ARTICLE_HEADING_SELECTOR = "h2, h3, h4"
ARTICLE_STOP_HEADINGS = {"関連記事", "コメント", "最近の記事", "カテゴリー", "アーカイブ"}
PROMO_PHRASES = [
    "Shisha Cafe & Bar LAGOS",
    "お客様ひとりひとりに合わせて",
    "店主のX",
    "気軽にお立ちよりくださいませ",
]
EMPTY_DUPLICATE_COLUMNS = [
    "kept_article_id",
    "removed_article_id",
    "duplicate_reason",
    "kept_url",
    "removed_url",
]
RECOMMENDED_HEADING_PATTERNS = [
    "おすすめミックス",
    "おすすめのミックス",
    "オススメミックス",
    "相性の良いフレーバー",
    "相性が良いフレーバー",
    "ミックス例",
    "おすすめブレンド",
]


@dataclass
class CategoryArticle:
    article_url: str
    article_title: str
    listed_date: str
    description: str


@dataclass
class FetchResult:
    url: str
    final_url: str
    status_code: int
    text: str
    content_type: str
    encoding: str
    retrieved_at: str
    sha256: str
    from_cache: bool


@dataclass
class RobotsCheckResult:
    robots_txt_url: str
    status_code: int
    allowed: bool
    matched_user_agent: str
    note: str


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text or ""))
    normalized = normalized.replace("オススメ", "おすすめ")
    normalized = normalized.lower()
    normalized = re.sub(r"[\s\u3000]+", "", normalized)
    normalized = re.sub(r"[|｜・,，。．!！?？:：()（）「」『』【】\[\]\"'`~〜\-ー_]+", "", normalized)
    return normalized


def is_recommended_mix_heading(text: str) -> bool:
    normalized = normalize_text(text)
    return any(normalize_text(pattern) in normalized for pattern in RECOMMENDED_HEADING_PATTERNS)


def ensure_allowed_domain(url: str, allowed_domain: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"unsupported scheme: {url}")
    if parsed.netloc and parsed.netloc != allowed_domain:
        raise ValueError(f"outside allowed domain: {url}")
    return url


def normalize_article_title(title: str) -> str:
    text = unicodedata.normalize("NFKC", str(title or ""))
    return re.sub(r"\s+", "", text).strip().lower()


def safe_slug_from_url(url: str) -> str:
    path = urlparse(url).path.rstrip("/")
    if not path:
        return "root"
    slug = path.split("/")[-1]
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", slug)
    return slug.strip("-") or hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


def url_to_storage_name(url: str) -> str:
    slug = safe_slug_from_url(url)
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
    return f"{slug}-{digest}"


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_brand_candidates(master_csv: Path) -> list[str]:
    if not master_csv.exists():
        return []
    df = pd.read_csv(master_csv, encoding="utf-8-sig")
    if "ブランド" not in df.columns:
        return []
    brands = sorted(
        {
            unicodedata.normalize("NFKC", str(value)).strip()
            for value in df["ブランド"].dropna().tolist()
            if str(value).strip()
        },
        key=lambda value: (-len(value), value),
    )
    return brands


def ascii_title_case(value: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9]+(?: [A-Za-z0-9]+)*", value):
        return " ".join(part[:1].upper() + part[1:].lower() for part in value.split())
    return value


def expand_brand_aliases(raw_brand: str) -> list[str]:
    value = unicodedata.normalize("NFKC", str(raw_brand or "")).strip()
    if not value:
        return []

    aliases: list[str] = []
    aliases.append(value)

    value_core = re.split(r"\s+シーシャフレーバー|\s+50g|\s+50ｇ|\s+100g|\s+100ｇ", value, maxsplit=1)[0].strip()
    if value_core:
        aliases.append(value_core)

    paren_stripped = re.sub(r"\s*[\(（].*?[\)）]", "", value_core).strip()
    if paren_stripped:
        aliases.append(paren_stripped)

    ascii_match = re.match(r"([A-Za-z]+(?:\s+[A-Za-z]+)*)", value_core)
    if ascii_match:
        ascii_alias = re.sub(r"\s+", " ", ascii_match.group(1).strip())
        if ascii_alias:
            aliases.append(ascii_alias)
            aliases.append(ascii_title_case(ascii_alias.upper()))

    unique: list[str] = []
    seen: set[str] = set()
    for alias in sorted((clean_text(alias) for alias in aliases if clean_text(alias)), key=len, reverse=True):
        key = alias.casefold()
        if key in seen:
            continue
        seen.add(key)
        unique.append(alias)
    return unique


def extract_brand_match(
    title: str,
    brand_candidates: list[str],
) -> tuple[str, str, str]:
    base = unicodedata.normalize("NFKC", str(title or "")).strip()
    base = base.split("|", 1)[0].strip()
    match = re.match(r"^(?P<prefix>.+?)のフレーバーレビュー", base)
    if not match:
        return "", "", ""
    prefix = match.group("prefix").strip()
    for brand in brand_candidates:
        for alias in expand_brand_aliases(brand):
            if prefix.casefold().startswith(alias.casefold()):
                flavor = prefix[len(alias):].strip(" 　-–—")
                return ascii_title_case(alias), flavor, brand
    return "", "", ""


def extract_brand_and_target_flavor(title: str, brand_candidates: list[str]) -> tuple[str, str]:
    brand, target_flavor, _matched_source = extract_brand_match(title, brand_candidates)
    return brand, target_flavor


def clean_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def block_text(block: Tag) -> str:
    if block.name == "table":
        rows = []
        for tr in block.select("tr"):
            cells = [clean_text(cell.get_text(" ", strip=True)) for cell in tr.select("th, td")]
            cells = [cell for cell in cells if cell]
            if cells:
                rows.append(" / ".join(cells))
        return "\n".join(rows).strip()
    if block.name in {"ul", "ol"}:
        items = [clean_text(li.get_text(" ", strip=True)) for li in block.select("li")]
        items = [item for item in items if item]
        return "\n".join(items).strip()
    return clean_text(block.get_text(" ", strip=True))


def should_skip_block(block: Tag, text: str) -> bool:
    if not text:
        return True
    if block.name == "blockquote" and "wp-embedded-content" in (block.get("class") or []):
        return True
    normalized = normalize_text(text)
    if normalized in {normalize_text(value) for value in ARTICLE_STOP_HEADINGS}:
        return True
    return any(phrase in text for phrase in PROMO_PHRASES)


def find_article_content_root(soup: BeautifulSoup) -> Tag | None:
    for selector in ARTICLE_CONTAINER_SELECTORS:
        node = soup.select_one(selector)
        if node is not None:
            return node
    return None


def find_article_root(soup: BeautifulSoup, content_root: Tag | None) -> Tag | None:
    for selector in ARTICLE_ROOT_SELECTORS:
        node = soup.select_one(selector)
        if node is not None:
            return node
    return content_root.parent if content_root is not None else None


def extract_author(soup: BeautifulSoup, article_root: Tag | None) -> str:
    meta_author = soup.select_one('meta[name="author"]')
    if meta_author is not None and meta_author.get("content"):
        return clean_text(meta_author["content"])

    if article_root is not None:
        author_node = (
            article_root.select_one(".author")
            or article_root.select_one(".post_author")
            or article_root.select_one('a[rel="author"]')
        )
        if author_node is not None:
            return clean_text(author_node.get_text(" ", strip=True))

    for script in soup.select('script[type="application/ld+json"]'):
        raw_json = clean_text(script.string or script.get_text(" ", strip=True))
        if not raw_json:
            continue
        try:
            payload = json.loads(raw_json)
        except json.JSONDecodeError:
            continue
        graph = payload.get("@graph") if isinstance(payload, dict) else None
        if not isinstance(graph, list):
            continue
        for node in graph:
            if not isinstance(node, dict):
                continue
            if node.get("@type") == "Person" and node.get("name"):
                return clean_text(node["name"])
    return ""


def parse_category_page(html: str, page_url: str, allowed_domain: str) -> tuple[list[CategoryArticle], str | None]:
    soup = BeautifulSoup(html, "html.parser")
    found: list[CategoryArticle] = []
    seen_urls: set[str] = set()
    for container in soup.select(LIST_CONTAINER_SELECTOR):
        for article in container.select(LIST_ITEM_SELECTOR):
            link = article.select_one(LIST_LINK_SELECTOR)
            if link is None or not link.get("href"):
                continue
            article_url = urljoin(page_url, link["href"])
            try:
                ensure_allowed_domain(article_url, allowed_domain)
            except ValueError:
                continue
            if article_url in seen_urls:
                continue
            seen_urls.add(article_url)
            title = clean_text(link.get_text(" ", strip=True))
            date_node = article.select_one(LIST_DATE_SELECTOR)
            desc_node = article.select_one(LIST_DESC_SELECTOR)
            found.append(
                CategoryArticle(
                    article_url=article_url,
                    article_title=title,
                    listed_date=clean_text(date_node.get_text(" ", strip=True)) if date_node else "",
                    description=clean_text(desc_node.get_text(" ", strip=True)) if desc_node else "",
                )
            )
    next_node = soup.select_one(NEXT_PAGE_SELECTOR)
    next_url = None
    if next_node is not None and next_node.get("href"):
        candidate = urljoin(page_url, next_node["href"])
        try:
            next_url = ensure_allowed_domain(candidate, allowed_domain)
        except ValueError:
            next_url = None
    return found, next_url


def parse_article_page(
    html: str,
    article_url: str,
    listed_title: str = "",
    listed_description: str = "",
    listed_date: str = "",
    category_label: str = CATEGORY_LABEL,
    brand_candidates: list[str] | None = None,
) -> dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    content_root = find_article_content_root(soup)
    article_root = find_article_root(soup, content_root)
    title_node = article_root.select_one("h1") if article_root is not None else None
    article_title = clean_text(title_node.get_text(" ", strip=True)) if title_node else clean_text(listed_title)
    canonical_node = soup.select_one('link[rel="canonical"]')
    canonical_url = clean_text(canonical_node["href"]) if canonical_node and canonical_node.get("href") else article_url
    published_node = article_root.select_one("time.entry-date.published") if article_root is not None else None
    updated_node = article_root.select_one("time.entry-date.updated") if article_root is not None else None
    headings: list[dict[str, Any]] = []
    paragraph_records: list[dict[str, Any]] = []
    table_records: list[dict[str, Any]] = []
    list_item_records: list[dict[str, Any]] = []
    recommended_mix_heading = ""
    recommended_mix_paragraphs: list[dict[str, Any]] = []
    paragraph_index = 1
    table_index = 1
    list_index = 1
    current_heading = ""
    current_heading_level = 0
    current_is_recommended = False

    if content_root is not None:
        for child in content_root.find_all(recursive=False):
            child_text = block_text(child)
            if child.name == "blockquote" and "wp-embedded-content" in (child.get("class") or []):
                break
            if should_skip_block(child, child_text):
                continue
            if child.name in {"h2", "h3", "h4"}:
                current_heading = child_text
                current_heading_level = int(child.name[1])
                current_is_recommended = is_recommended_mix_heading(current_heading)
                headings.append({"heading_level": current_heading_level, "heading_text": current_heading})
                if current_is_recommended and not recommended_mix_heading:
                    recommended_mix_heading = current_heading
                continue

            if child.name == "table":
                for row_index, tr in enumerate(child.select("tr"), start=1):
                    cells = [clean_text(cell.get_text(" ", strip=True)) for cell in tr.select("th, td")]
                    for cell_index, cell_text in enumerate(cells, start=1):
                        if not cell_text:
                            continue
                        table_records.append(
                            {
                                "table_index": table_index,
                                "row_index": row_index,
                                "cell_index": cell_index,
                                "section_heading": current_heading,
                                "cell_text": cell_text,
                                "is_recommended_mix_section": current_is_recommended,
                            }
                        )
                table_index += 1

            if child.name in {"ul", "ol"}:
                for item_index, li in enumerate(child.select("li"), start=1):
                    item_text = clean_text(li.get_text(" ", strip=True))
                    if not item_text:
                        continue
                    list_item_records.append(
                        {
                            "list_index": list_index,
                            "item_index": item_index,
                            "section_heading": current_heading,
                            "item_text": item_text,
                            "is_recommended_mix_section": current_is_recommended,
                        }
                    )
                list_index += 1

            block_lines = [line for line in child_text.split("\n") if line.strip()]
            if not block_lines:
                continue
            for line in block_lines:
                if any(phrase in line for phrase in PROMO_PHRASES):
                    continue
                record = {
                    "paragraph_index": paragraph_index,
                    "heading_level": current_heading_level if current_heading else "",
                    "section_heading": current_heading,
                    "paragraph_text": clean_text(line),
                    "is_recommended_mix_section": current_is_recommended,
                }
                paragraph_records.append(record)
                if current_is_recommended:
                    recommended_mix_paragraphs.append(record)
                paragraph_index += 1

    main_text = "\n".join(record["paragraph_text"] for record in paragraph_records).strip()
    brand, target_flavor, matched_brand_source = extract_brand_match(article_title, brand_candidates or [])
    recommended_mix_text = "\n".join(record["paragraph_text"] for record in recommended_mix_paragraphs).strip()
    return {
        "article_url": article_url,
        "canonical_url": canonical_url,
        "article_slug": safe_slug_from_url(canonical_url or article_url),
        "article_title": article_title,
        "published_date": clean_text(published_node.get_text(" ", strip=True)) if published_node else clean_text(listed_date),
        "updated_date": clean_text(updated_node.get_text(" ", strip=True)) if updated_node else "",
        "author": extract_author(soup, article_root),
        "category": category_label,
        "description": clean_text(listed_description),
        "main_text": main_text,
        "headings": headings,
        "paragraphs": paragraph_records,
        "table_records": table_records,
        "list_item_records": list_item_records,
        "recommended_mix_section_found": bool(recommended_mix_heading),
        "recommended_mix_heading": recommended_mix_heading,
        "recommended_mix_text": recommended_mix_text,
        "recommended_mix_paragraphs": recommended_mix_paragraphs,
        "recommended_mix_paragraph_count": len(recommended_mix_paragraphs),
        "brand": brand,
        "target_flavor": target_flavor,
        "matched_brand_dictionary_value": matched_brand_source,
        "source_site": SOURCE_SITE,
        "source_type": SOURCE_TYPE,
        "is_empty_body": not bool(main_text),
    }


class RawHtmlStore:
    def __init__(self, raw_dir: Path) -> None:
        self.raw_dir = raw_dir

    def _paths(self, url: str, page_type: str) -> tuple[Path, Path]:
        subdir = self.raw_dir / page_type
        subdir.mkdir(parents=True, exist_ok=True)
        stem = url_to_storage_name(url)
        return subdir / f"{stem}.html", subdir / f"{stem}.json"

    def load(self, url: str, page_type: str) -> FetchResult | None:
        html_path, meta_path = self._paths(url, page_type)
        if not html_path.exists() or not meta_path.exists():
            return None
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        text = html_path.read_text(encoding="utf-8")
        return FetchResult(
            url=url,
            final_url=meta.get("final_url", url),
            status_code=int(meta.get("status_code", 200)),
            text=text,
            content_type=meta.get("content_type", "text/html"),
            encoding=meta.get("encoding", "utf-8"),
            retrieved_at=meta.get("retrieved_at", ""),
            sha256=meta.get("sha256", sha256_hex(text)),
            from_cache=True,
        )

    def save(self, url: str, page_type: str, result: FetchResult) -> None:
        html_path, meta_path = self._paths(url, page_type)
        html_path.write_text(result.text, encoding="utf-8")
        meta = {
            "url": url,
            "final_url": result.final_url,
            "status_code": result.status_code,
            "retrieved_at": result.retrieved_at,
            "content_type": result.content_type,
            "encoding": result.encoding,
            "sha256": result.sha256,
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class SafeHttpClient:
    def __init__(
        self,
        user_agent: str,
        allowed_domain: str,
        delay: float = 2.0,
        jitter: float = 1.0,
        timeout: float = 20.0,
        retries: int = 3,
        backoff_factor: float = 2.0,
        max_response_bytes: int = 2_000_000,
        session: requests.Session | None = None,
        sleep_func: Any = time.sleep,
        random_func: Any = random.uniform,
        logger: logging.Logger | None = None,
        raw_store: RawHtmlStore | None = None,
        resume: bool = False,
    ) -> None:
        self.user_agent = user_agent
        self.allowed_domain = allowed_domain
        self.delay = delay
        self.jitter = jitter
        self.timeout = timeout
        self.retries = retries
        self.backoff_factor = backoff_factor
        self.max_response_bytes = max_response_bytes
        self.session = session or requests.Session()
        self.sleep_func = sleep_func
        self.random_func = random_func
        self.logger = logger or logging.getLogger(__name__)
        self.raw_store = raw_store
        self.resume = resume

    def _delay(self) -> None:
        span = max(0.0, self.jitter)
        self.sleep_func(self.delay + (self.random_func(0.0, span) if span else 0.0))

    def fetch(self, url: str, page_type: str) -> FetchResult:
        ensure_allowed_domain(url, self.allowed_domain)
        if self.resume and self.raw_store is not None:
            cached = self.raw_store.load(url, page_type)
            if cached is not None:
                self.logger.info("resume cache hit: %s", url)
                return cached

        headers = {"User-Agent": self.user_agent, "Accept": "text/html,application/xhtml+xml"}
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 2):
            if attempt > 1:
                sleep_seconds = self.backoff_factor ** (attempt - 2)
                self.logger.warning("retrying %s after %.1fs (attempt %s)", url, sleep_seconds, attempt)
                self.sleep_func(sleep_seconds)
            self._delay()
            try:
                response = self.session.get(url, headers=headers, timeout=self.timeout, stream=True)
                status_code = response.status_code
                content_type = response.headers.get("Content-Type", "")
                if status_code in {429, 500, 502, 503, 504}:
                    if attempt <= self.retries + 1:
                        last_error = RuntimeError(f"retryable status {status_code}")
                        continue
                if status_code == 404:
                    raise FileNotFoundError(f"404 not found: {url}")
                response.raise_for_status()
                if "html" not in content_type.lower():
                    raise ValueError(f"unexpected content type: {content_type}")
                chunks: list[bytes] = []
                total = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > self.max_response_bytes:
                        raise ValueError(f"response too large: {total}")
                    chunks.append(chunk)
                body = b"".join(chunks)
                encoding = response.encoding or response.apparent_encoding or "utf-8"
                text = body.decode(encoding, errors="replace")
                result = FetchResult(
                    url=url,
                    final_url=response.url,
                    status_code=status_code,
                    text=text,
                    content_type=content_type,
                    encoding=encoding,
                    retrieved_at=utc_now_iso(),
                    sha256=hashlib.sha256(body).hexdigest(),
                    from_cache=False,
                )
                if self.raw_store is not None:
                    self.raw_store.save(url, page_type, result)
                return result
            except FileNotFoundError:
                raise
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt > self.retries:
                    break
        raise RuntimeError(f"failed to fetch {url}: {last_error}") from last_error


def configure_logger(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("shisha_lagos_scraper")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level.upper())
    return logger


def default_user_agent(contact: str = "") -> str:
    base = "ShishaResearchBot/1.0 (+research scraping for JCEEE paper)"
    if contact:
        return f"{base} contact={contact}"
    return base


def check_robots_txt(
    start_url: str,
    user_agent: str,
    session: requests.Session | None = None,
    timeout: float = 20.0,
) -> RobotsCheckResult:
    parsed = urlparse(start_url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    active_session = session or requests.Session()
    response = active_session.get(robots_url, headers={"User-Agent": user_agent}, timeout=timeout)
    content = response.text
    parser = RobotFileParser()
    parser.set_url(robots_url)
    parser.parse(content.splitlines())
    allowed = parser.can_fetch(user_agent, start_url)
    note = "allowed" if allowed else "disallowed by robots.txt"
    return RobotsCheckResult(
        robots_txt_url=robots_url,
        status_code=response.status_code,
        allowed=allowed,
        matched_user_agent=user_agent,
        note=note,
    )


def article_id_from_slug(slug: str) -> str:
    return f"lagos_{slug}"


def deduplicate_articles(
    articles_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if articles_df.empty:
        return articles_df.copy(), pd.DataFrame(columns=EMPTY_DUPLICATE_COLUMNS)
    df = articles_df.copy().reset_index(drop=True)
    df["normalized_title"] = df["article_title"].map(normalize_article_title)
    df["text_sha256"] = df["main_text"].map(sha256_hex)
    keep_indices: set[int] = set()
    duplicate_rows: list[dict[str, Any]] = []
    seen: dict[tuple[str, str], int] = {}
    reason_columns = [
        ("article_url", "url_exact_match"),
        ("canonical_url", "canonical_url_match"),
        ("article_slug", "slug_match"),
        ("normalized_title", "normalized_title_match"),
        ("text_sha256", "main_text_hash_match"),
    ]

    for idx, row in df.iterrows():
        matched_index = None
        matched_reason = ""
        for column, reason in reason_columns:
            value = str(row.get(column, "") or "").strip()
            if not value:
                continue
            key = (column, value)
            if key in seen:
                matched_index = seen[key]
                matched_reason = reason
                break
        if matched_index is None:
            keep_indices.add(idx)
            for column, _reason in reason_columns:
                value = str(row.get(column, "") or "").strip()
                if value:
                    seen[(column, value)] = idx
            continue
        kept_row = df.loc[matched_index]
        duplicate_rows.append(
            {
                "kept_article_id": kept_row["article_id"],
                "removed_article_id": row["article_id"],
                "duplicate_reason": matched_reason,
                "kept_url": kept_row["article_url"],
                "removed_url": row["article_url"],
            }
        )

    deduped = df.loc[sorted(keep_indices)].drop(columns=["normalized_title", "text_sha256"]).reset_index(drop=True)
    duplicates_df = pd.DataFrame(duplicate_rows, columns=EMPTY_DUPLICATE_COLUMNS)
    return deduped, duplicates_df


def build_paragraph_dataframe(articles: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for article in articles:
        for paragraph in article["paragraphs"]:
            rows.append(
                {
                    "article_id": article["article_id"],
                    "article_url": article["article_url"],
                    "article_title": article["article_title"],
                    "paragraph_index": paragraph["paragraph_index"],
                    "heading_level": paragraph["heading_level"],
                    "section_heading": paragraph["section_heading"],
                    "paragraph_text": paragraph["paragraph_text"],
                    "is_recommended_mix_section": paragraph["is_recommended_mix_section"],
                }
            )
    return pd.DataFrame(rows)


def build_tables_dataframe(articles: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for article in articles:
        for cell in article.get("table_records", []):
            rows.append(
                {
                    "article_id": article["article_id"],
                    "article_url": article["article_url"],
                    "table_index": cell["table_index"],
                    "row_index": cell["row_index"],
                    "cell_index": cell["cell_index"],
                    "section_heading": cell["section_heading"],
                    "cell_text": cell["cell_text"],
                    "is_recommended_mix_section": cell["is_recommended_mix_section"],
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "article_id",
            "article_url",
            "table_index",
            "row_index",
            "cell_index",
            "section_heading",
            "cell_text",
            "is_recommended_mix_section",
        ],
    )


def build_list_items_dataframe(articles: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for article in articles:
        for item in article.get("list_item_records", []):
            rows.append(
                {
                    "article_id": article["article_id"],
                    "article_url": article["article_url"],
                    "list_index": item["list_index"],
                    "item_index": item["item_index"],
                    "section_heading": item["section_heading"],
                    "item_text": item["item_text"],
                    "is_recommended_mix_section": item["is_recommended_mix_section"],
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "article_id",
            "article_url",
            "list_index",
            "item_index",
            "section_heading",
            "item_text",
            "is_recommended_mix_section",
        ],
    )


def build_recommended_mix_dataframe(articles: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for article in articles:
        rows.append(
            {
                "article_id": article["article_id"],
                "article_url": article["article_url"],
                "article_title": article["article_title"],
                "target_flavor": article["target_flavor"],
                "recommended_mix_heading": article["recommended_mix_heading"],
                "recommended_mix_text": article["recommended_mix_text"],
                "recommended_mix_paragraph_count": article["recommended_mix_paragraph_count"],
            }
        )
    return pd.DataFrame(rows)


def build_articles_dataframe(articles: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for article in articles:
        rows.append(
            {
                "article_id": article["article_id"],
                "source_site": article["source_site"],
                "source_type": article["source_type"],
                "category": article["category"],
                "article_url": article["article_url"],
                "canonical_url": article["canonical_url"],
                "article_slug": article["article_slug"],
                "article_title": article["article_title"],
                "brand": article["brand"],
                "target_flavor": article["target_flavor"],
                "matched_brand_dictionary_value": article["matched_brand_dictionary_value"],
                "published_date": article["published_date"],
                "updated_date": article["updated_date"],
                "author": article["author"],
                "description": article["description"],
                "main_text": article["main_text"],
                "headings": json.dumps(article["headings"], ensure_ascii=False),
                "paragraphs": json.dumps(article["paragraphs"], ensure_ascii=False),
                "recommended_mix_section_found": article["recommended_mix_section_found"],
                "recommended_mix_heading": article["recommended_mix_heading"],
                "recommended_mix_text": article["recommended_mix_text"],
                "scraped_at": article["scraped_at"],
                "html_sha256": article["html_sha256"],
            }
        )
    return pd.DataFrame(rows)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def summarize_articles(
    category_page_count: int,
    discovered_articles: list[CategoryArticle],
    article_records: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    duplicates_df: pd.DataFrame,
) -> dict[str, Any]:
    main_lengths = [len(record["main_text"]) for record in article_records if record["main_text"]]
    empty_body_count = sum(1 for record in article_records if not record["main_text"])
    target_flavor_success = sum(1 for record in article_records if record["target_flavor"])
    published_success = sum(1 for record in article_records if record["published_date"])
    recommended_found = sum(1 for record in article_records if record["recommended_mix_section_found"])
    tail_keywords = ["最近の記事", "カテゴリー", "アーカイブ", "Shisha Cafe & Bar LAGOS"]
    contamination_count = sum(
        1
        for record in article_records
        if any(keyword in record["main_text"][-300:] for keyword in tail_keywords)
    )
    return {
        "category_page_count": category_page_count,
        "article_url_count": len(discovered_articles),
        "successful_article_count": len(article_records),
        "failed_article_count": len(failures),
        "duplicate_article_count": int(len(duplicates_df)),
        "empty_body_article_count": empty_body_count,
        "main_text_length_min": min(main_lengths) if main_lengths else 0,
        "main_text_length_median": float(pd.Series(main_lengths).median()) if main_lengths else 0,
        "main_text_length_max": max(main_lengths) if main_lengths else 0,
        "recommended_mix_section_found_count": recommended_found,
        "target_flavor_extracted_count": target_flavor_success,
        "published_date_extracted_count": published_success,
        "domain_outside_count": 0,
        "suspected_sidebar_or_footer_contamination_count": contamination_count,
    }


def write_quality_outputs(
    report_dir: Path,
    summary: dict[str, Any],
    article_records: list[dict[str, Any]],
) -> tuple[Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = report_dir / "shisha_lagos_scraping_summary.csv"
    report_md = report_dir / "shisha_lagos_scraping_report.md"
    write_csv(pd.DataFrame([summary]), summary_csv)

    sample_rows = article_records[:3]
    lines = [
        "# Shisha Lagos Scraping Report",
        "",
        "## Summary",
        "",
    ]
    for key, value in summary.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Sample Articles", ""])
    for article in sample_rows:
        headings = [heading["heading_text"] for heading in article["headings"]]
        lines.extend(
            [
                f"### {article['article_title']}",
                "",
                f"- URL: {article['article_url']}",
                f"- 本文文字数: {len(article['main_text'])}",
                f"- 見出し: {', '.join(headings) if headings else 'なし'}",
                f"- おすすめミックス節: {'あり' if article['recommended_mix_section_found'] else 'なし'}",
                "",
                "本文先頭300文字:",
                article["main_text"][:300],
                "",
                "本文末尾300文字:",
                article["main_text"][-300:],
                "",
            ]
        )
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_csv, report_md


def build_metadata(
    start_url: str,
    git_commit: str,
    delay: float,
    user_agent: str,
    max_pages: int,
    summary: dict[str, Any],
    robots: RobotsCheckResult,
    output_files: list[str],
) -> dict[str, Any]:
    return {
        "start_url": start_url,
        "source_site": SOURCE_SITE,
        "source_type": SOURCE_TYPE,
        "scraped_at": utc_now_iso(),
        "scraper_version": SCRAPER_VERSION,
        "git_commit": git_commit,
        "request_delay": delay,
        "user_agent": user_agent,
        "max_pages": max_pages,
        "category_page_count": summary["category_page_count"],
        "article_url_count": summary["article_url_count"],
        "successful_article_count": summary["successful_article_count"],
        "failed_article_count": summary["failed_article_count"],
        "robots_txt_url": robots.robots_txt_url,
        "robots_txt_status": robots.status_code,
        "output_files": output_files,
    }


def write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def git_commit_hash(repo_dir: Path) -> str:
    try:
        import subprocess

        return subprocess.check_output(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:  # noqa: BLE001
        return ""


def scrape_category_and_articles(
    start_url: str,
    client: SafeHttpClient,
    max_pages: int,
    dry_run: bool,
    max_article_fetches: int = 3,
    brand_candidates: list[str] | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    active_logger = logger or logging.getLogger(__name__)
    visited_pages: set[str] = set()
    article_map: dict[str, CategoryArticle] = {}
    category_page_count = 0
    next_url = start_url

    while next_url and category_page_count < max_pages:
        if next_url in visited_pages:
            active_logger.warning("skip already visited page: %s", next_url)
            break
        visited_pages.add(next_url)
        page_result = client.fetch(next_url, "category_pages")
        category_page_count += 1
        page_articles, candidate_next = parse_category_page(page_result.text, next_url, client.allowed_domain)
        for article in page_articles:
            article_map.setdefault(article.article_url, article)
        next_url = candidate_next

    sorted_articles = list(article_map.values())
    article_fetch_limit = min(len(sorted_articles), max_article_fetches) if dry_run else len(sorted_articles)
    fetched_records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for article in sorted_articles[:article_fetch_limit]:
        try:
            result = client.fetch(article.article_url, "articles")
            parsed = parse_article_page(
                result.text,
                article.article_url,
                listed_title=article.article_title,
                listed_description=article.description,
                listed_date=article.listed_date,
                brand_candidates=brand_candidates or [],
            )
            parsed["article_id"] = article_id_from_slug(parsed["article_slug"])
            parsed["scraped_at"] = result.retrieved_at
            parsed["html_sha256"] = result.sha256
            fetched_records.append(parsed)
        except Exception as exc:  # noqa: BLE001
            failures.append({"article_url": article.article_url, "error": str(exc)})
            active_logger.warning("failed article: %s (%s)", article.article_url, exc)

    articles_df = build_articles_dataframe(fetched_records)
    deduped_articles_df, duplicates_df = deduplicate_articles(articles_df)
    kept_ids = set(deduped_articles_df["article_id"].tolist()) if not deduped_articles_df.empty else set()
    deduped_records = [record for record in fetched_records if record["article_id"] in kept_ids]
    paragraph_df = build_paragraph_dataframe(deduped_records)
    tables_df = build_tables_dataframe(deduped_records)
    list_items_df = build_list_items_dataframe(deduped_records)
    recommended_df = build_recommended_mix_dataframe(deduped_records)
    summary = summarize_articles(category_page_count, sorted_articles, deduped_records, failures, duplicates_df)
    return {
        "category_page_count": category_page_count,
        "discovered_articles": sorted_articles,
        "fetched_records": deduped_records,
        "failures": failures,
        "articles_df": deduped_articles_df,
        "paragraph_df": paragraph_df,
        "tables_df": tables_df,
        "list_items_df": list_items_df,
        "recommended_df": recommended_df,
        "duplicates_df": duplicates_df,
        "summary": summary,
    }


def dry_run_output(scrape_result: dict[str, Any]) -> str:
    lines = [
        "dry-run summary",
        f"- category pages checked: {scrape_result['category_page_count']}",
        f"- discovered article urls: {len(scrape_result['discovered_articles'])}",
        f"- fetched preview articles: {len(scrape_result['fetched_records'])}",
    ]
    for record in scrape_result["fetched_records"]:
        headings = ", ".join(heading["heading_text"] for heading in record["headings"][:5])
        lines.extend(
            [
                f"  * {record['article_title']}",
                f"    - url: {record['article_url']}",
                f"    - main_text_length: {len(record['main_text'])}",
                f"    - headings: {headings}",
                f"    - recommended_mix_section_found: {record['recommended_mix_section_found']}",
            ]
        )
    return "\n".join(lines)
