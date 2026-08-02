#!/usr/bin/env python3
"""Tests for Shisha LAGOS scraper."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import sys

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from shisha_lagos_scraper import (  # noqa: E402
    FetchResult,
    RawHtmlStore,
    SafeHttpClient,
    article_id_from_slug,
    build_articles_dataframe,
    deduplicate_articles,
    extract_brand_and_target_flavor,
    is_recommended_mix_heading,
    parse_article_page,
    parse_category_page,
    scrape_category_and_articles,
)


class FakeResponse:
    def __init__(
        self,
        status_code: int = 200,
        text: str = "",
        url: str = "https://shisha-lagos.com/test/",
        headers: dict[str, str] | None = None,
        encoding: str = "utf-8",
    ) -> None:
        self.status_code = status_code
        self._text = text
        self.url = url
        self.headers = headers or {"Content-Type": "text/html; charset=utf-8"}
        self.encoding = encoding
        self.apparent_encoding = encoding

    def iter_content(self, chunk_size: int = 8192):  # noqa: ARG002
        yield self._text.encode(self.encoding)

    @property
    def text(self) -> str:
        return self._text

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"status={self.status_code}")


class FakeSession:
    def __init__(self, responses: list[FakeResponse | Exception]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    def get(self, url: str, **kwargs):  # noqa: ANN003
        self.calls.append(url)
        if not self.responses:
            raise AssertionError("no more fake responses")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeClient:
    def __init__(self, category_html: str, article_html: str, allowed_domain: str = "shisha-lagos.com") -> None:
        self.allowed_domain = allowed_domain
        self.category_html = category_html
        self.article_html = article_html
        self.calls: list[tuple[str, str]] = []

    def fetch(self, url: str, page_type: str) -> FetchResult:
        self.calls.append((url, page_type))
        if page_type == "category_pages":
            return FetchResult(
                url=url,
                final_url=url,
                status_code=200,
                text=self.category_html,
                content_type="text/html",
                encoding="utf-8",
                retrieved_at="2026-07-29T00:00:00+00:00",
                sha256="categorysha",
                from_cache=False,
            )
        if "notfound" in url:
            raise FileNotFoundError(url)
        return FetchResult(
            url=url,
            final_url=url,
            status_code=200,
            text=self.article_html,
            content_type="text/html",
            encoding="utf-8",
            retrieved_at="2026-07-29T00:00:00+00:00",
            sha256="articlesha",
            from_cache=False,
        )


class ShishaLagosScraperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        fixture_dir = Path(__file__).resolve().parent / "fixtures" / "shisha_lagos"
        cls.category_html = (fixture_dir / "category_page_1.html").read_text(encoding="utf-8")
        cls.article_html = (fixture_dir / "article_example.html").read_text(encoding="utf-8")
        cls.brand_candidates = ["AL FAKHER（アルファーヘル） シーシャフレーバー 50g", "Azure"]

    def test_category_parser_uses_blog_list_only(self) -> None:
        articles, next_url = parse_category_page(
            self.category_html,
            "https://shisha-lagos.com/category/%E3%83%95%E3%83%AC%E3%83%BC%E3%83%90%E3%83%BC%E3%83%AC%E3%83%93%E3%83%A5%E3%83%BC/",
            "shisha-lagos.com",
        )
        self.assertEqual(len(articles), 2)
        self.assertEqual(articles[0].article_url, "https://shisha-lagos.com/af-watermelon-review/")
        self.assertEqual(articles[0].listed_date, "2025.12.10")
        self.assertEqual(
            next_url,
            "https://shisha-lagos.com/category/%E3%83%95%E3%83%AC%E3%83%BC%E3%83%90%E3%83%BC%E3%83%AC%E3%83%93%E3%83%A5%E3%83%BC/page/2/",
        )

    def test_category_parser_stops_when_no_next(self) -> None:
        html = self.category_html.replace(
            '<div class="page_navi clearfix">\n    <ul class="page-numbers">\n      <li><a class="next page-numbers" href="https://shisha-lagos.com/category/%E3%83%95%E3%83%AC%E3%83%BC%E3%83%90%E3%83%BC%E3%83%AC%E3%83%93%E3%83%A5%E3%83%BC/page/2/"><span>»</span></a></li>\n    </ul>\n  </div>',
            "",
        )
        _articles, next_url = parse_category_page(
            html,
            "https://shisha-lagos.com/category/%E3%83%95%E3%83%AC%E3%83%BC%E3%83%90%E3%83%BC%E3%83%AC%E3%83%93%E3%83%A5%E3%83%BC/",
            "shisha-lagos.com",
        )
        self.assertIsNone(next_url)

    def test_article_parser_extracts_content_and_recommended_mix(self) -> None:
        article = parse_article_page(
            self.article_html,
            "https://shisha-lagos.com/af-watermelon-review/",
            listed_title="Al Fakherスイカのフレーバーレビュー|特徴やおすすめミックスなど",
            listed_description="desc",
            listed_date="2025.12.10",
            brand_candidates=self.brand_candidates,
        )
        self.assertEqual(article["article_title"], "Al Fakherスイカのフレーバーレビュー|特徴やおすすめミックスなど")
        self.assertEqual(article["published_date"], "2025.12.10")
        self.assertEqual(article["updated_date"], "2025.01.09")
        self.assertEqual(article["brand"], "Al Fakher")
        self.assertEqual(article["target_flavor"], "スイカ")
        self.assertEqual(article["matched_brand_dictionary_value"], "AL FAKHER（アルファーヘル） シーシャフレーバー 50g")
        self.assertEqual(article["author"], "nkou1213")
        self.assertTrue(article["recommended_mix_section_found"])
        self.assertIn("レモンやミント", article["recommended_mix_text"])
        self.assertIn("メロンを加える", article["recommended_mix_text"])
        self.assertNotIn("Shisha Cafe & Bar LAGOS", article["main_text"])
        self.assertNotIn("関連記事", article["main_text"])
        self.assertGreater(len(article["table_records"]), 0)
        self.assertGreater(len(article["list_item_records"]), 0)
        self.assertEqual(article["paragraphs"][0]["section_heading"], "")
        self.assertEqual(article["paragraphs"][1]["section_heading"], "Al Fakherスイカの特徴")
        self.assertTrue(any(paragraph["is_recommended_mix_section"] for paragraph in article["paragraphs"]))

    def test_article_parser_detects_empty_body(self) -> None:
        html = "<html><body><div id='article'><h1>Title</h1></div></body></html>"
        article = parse_article_page(
            html,
            "https://shisha-lagos.com/empty/",
            listed_title="Title",
            brand_candidates=self.brand_candidates,
        )
        self.assertTrue(article["is_empty_body"])
        self.assertEqual(article["main_text"], "")

    def test_recommended_heading_normalization(self) -> None:
        self.assertTrue(is_recommended_mix_heading("オススメミックス"))
        self.assertTrue(is_recommended_mix_heading("相性が良いフレーバー"))
        self.assertFalse(is_recommended_mix_heading("特徴"))

    def test_target_flavor_extraction(self) -> None:
        brand, flavor = extract_brand_and_target_flavor(
            "Al Fakherスイカのフレーバーレビュー|特徴やおすすめミックスなど",
            self.brand_candidates,
        )
        self.assertEqual(brand, "Al Fakher")
        self.assertEqual(flavor, "スイカ")
        brand, flavor = extract_brand_and_target_flavor("不明タイトル", self.brand_candidates)
        self.assertEqual((brand, flavor), ("", ""))

    def test_http_retries_on_429(self) -> None:
        session = FakeSession(
            [
                FakeResponse(status_code=429, text="busy"),
                FakeResponse(status_code=200, text="<html>ok</html>"),
            ]
        )
        client = SafeHttpClient(
            user_agent="test-bot",
            allowed_domain="shisha-lagos.com",
            session=session,
            delay=0.0,
            jitter=0.0,
            retries=2,
            sleep_func=lambda _value: None,
        )
        result = client.fetch("https://shisha-lagos.com/test/", "articles")
        self.assertEqual(result.status_code, 200)
        self.assertEqual(len(session.calls), 2)

    def test_http_retries_on_5xx(self) -> None:
        session = FakeSession(
            [
                FakeResponse(status_code=503, text="busy"),
                FakeResponse(status_code=200, text="<html>ok</html>"),
            ]
        )
        client = SafeHttpClient(
            user_agent="test-bot",
            allowed_domain="shisha-lagos.com",
            session=session,
            delay=0.0,
            jitter=0.0,
            retries=2,
            sleep_func=lambda _value: None,
        )
        result = client.fetch("https://shisha-lagos.com/test/", "articles")
        self.assertEqual(result.status_code, 200)
        self.assertEqual(len(session.calls), 2)

    def test_http_404_is_raised(self) -> None:
        session = FakeSession([FakeResponse(status_code=404, text="missing")])
        client = SafeHttpClient(
            user_agent="test-bot",
            allowed_domain="shisha-lagos.com",
            session=session,
            delay=0.0,
            jitter=0.0,
            retries=0,
            sleep_func=lambda _value: None,
        )
        with self.assertRaises(FileNotFoundError):
            client.fetch("https://shisha-lagos.com/missing/", "articles")

    def test_http_rejects_outside_domain(self) -> None:
        session = FakeSession([FakeResponse(status_code=200, text="<html></html>")])
        client = SafeHttpClient(
            user_agent="test-bot",
            allowed_domain="shisha-lagos.com",
            session=session,
            delay=0.0,
            jitter=0.0,
        )
        with self.assertRaises(ValueError):
            client.fetch("https://example.com/outside/", "articles")

    def test_resume_uses_saved_html(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RawHtmlStore(Path(tmpdir))
            saved = FetchResult(
                url="https://shisha-lagos.com/test/",
                final_url="https://shisha-lagos.com/test/",
                status_code=200,
                text="<html>cached</html>",
                content_type="text/html",
                encoding="utf-8",
                retrieved_at="2026-07-29T00:00:00+00:00",
                sha256="abc",
                from_cache=False,
            )
            store.save("https://shisha-lagos.com/test/", "articles", saved)
            session = FakeSession([])
            client = SafeHttpClient(
                user_agent="test-bot",
                allowed_domain="shisha-lagos.com",
                session=session,
                delay=0.0,
                jitter=0.0,
                raw_store=store,
                resume=True,
            )
            result = client.fetch("https://shisha-lagos.com/test/", "articles")
            self.assertTrue(result.from_cache)
            self.assertEqual(result.text, "<html>cached</html>")
            self.assertEqual(len(session.calls), 0)

    def test_scrape_records_404_failure_and_skips(self) -> None:
        category_html = self.category_html.replace(
            "https://shisha-lagos.com/af-coconut-review/",
            "https://shisha-lagos.com/notfound/",
        )
        client = FakeClient(category_html, self.article_html)
        result = scrape_category_and_articles(
            start_url="https://shisha-lagos.com/category/%E3%83%95%E3%83%AC%E3%83%BC%E3%83%90%E3%83%BC%E3%83%AC%E3%83%93%E3%83%A5%E3%83%BC/",
            client=client,
            max_pages=1,
            dry_run=False,
            max_article_fetches=10,
            brand_candidates=self.brand_candidates,
        )
        self.assertEqual(result["summary"]["article_url_count"], 2)
        self.assertEqual(len(result["fetched_records"]), 1)
        self.assertEqual(len(result["failures"]), 1)

    def test_deduplicate_articles(self) -> None:
        records = [
            {
                "article_id": article_id_from_slug("af-watermelon-review"),
                "article_url": "https://shisha-lagos.com/af-watermelon-review/",
                "canonical_url": "https://shisha-lagos.com/af-watermelon-review/",
                "article_slug": "af-watermelon-review",
                "article_title": "Al Fakherスイカのフレーバーレビュー|特徴やおすすめミックスなど",
                "main_text": "same text",
                "source_site": "shisha_lagos",
                "source_type": "editorial_review",
                "category": "フレーバーレビュー",
                "published_date": "",
                "updated_date": "",
                "author": "",
                "description": "",
                "headings": "[]",
                "paragraphs": "[]",
                "recommended_mix_section_found": True,
                "recommended_mix_heading": "",
                "recommended_mix_text": "",
                "scraped_at": "",
                "html_sha256": "1",
                "brand": "Al Fakher",
                "target_flavor": "スイカ",
                "matched_brand_dictionary_value": "AL FAKHER（アルファーヘル） シーシャフレーバー 50g",
            },
            {
                "article_id": article_id_from_slug("af-watermelon-review-copy"),
                "article_url": "https://shisha-lagos.com/af-watermelon-review-copy/",
                "canonical_url": "https://shisha-lagos.com/af-watermelon-review/",
                "article_slug": "af-watermelon-review-copy",
                "article_title": "Al Fakherスイカのフレーバーレビュー|特徴やおすすめミックスなど",
                "main_text": "same text",
                "source_site": "shisha_lagos",
                "source_type": "editorial_review",
                "category": "フレーバーレビュー",
                "published_date": "",
                "updated_date": "",
                "author": "",
                "description": "",
                "headings": "[]",
                "paragraphs": "[]",
                "recommended_mix_section_found": True,
                "recommended_mix_heading": "",
                "recommended_mix_text": "",
                "scraped_at": "",
                "html_sha256": "2",
                "brand": "Al Fakher",
                "target_flavor": "スイカ",
                "matched_brand_dictionary_value": "AL FAKHER（アルファーヘル） シーシャフレーバー 50g",
            },
        ]
        articles_df = build_articles_dataframe(records)
        deduped_df, duplicates_df = deduplicate_articles(articles_df)
        self.assertEqual(len(deduped_df), 1)
        self.assertEqual(len(duplicates_df), 1)

    def test_empty_duplicate_dataframe_keeps_columns(self) -> None:
        empty = build_articles_dataframe([])
        deduped, duplicates = deduplicate_articles(empty)
        self.assertEqual(len(deduped), 0)
        self.assertEqual(
            duplicates.columns.tolist(),
            ["kept_article_id", "removed_article_id", "duplicate_reason", "kept_url", "removed_url"],
        )


if __name__ == "__main__":
    unittest.main()
