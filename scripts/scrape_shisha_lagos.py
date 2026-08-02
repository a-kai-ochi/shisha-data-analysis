#!/usr/bin/env python3
"""Scrape Shisha Cafe & Bar LAGOS flavor-review articles."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import requests

from shisha_lagos_scraper import (
    DEFAULT_ALLOWED_DOMAIN,
    DEFAULT_START_URL,
    RawHtmlStore,
    SafeHttpClient,
    SOURCE_SITE,
    SOURCE_TYPE,
    build_metadata,
    check_robots_txt,
    configure_logger,
    default_user_agent,
    dry_run_output,
    git_commit_hash,
    read_brand_candidates,
    scrape_category_and_articles,
    write_csv,
    write_metadata,
    write_quality_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-url", default=DEFAULT_START_URL)
    parser.add_argument("--output-dir", default=str(root / "data" / "processed"))
    parser.add_argument("--raw-dir", default=str(root / "data" / "raw" / SOURCE_SITE))
    parser.add_argument("--report-dir", default=str(root / "outputs"))
    parser.add_argument("--master-csv", default=str(root / "data" / "aslaj_master_list.csv"))
    parser.add_argument("--delay", type=float, default=2.0)
    parser.add_argument("--jitter", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--backoff-factor", type=float, default=2.0)
    parser.add_argument("--max-response-bytes", type=int, default=2_000_000)
    parser.add_argument("--max-pages", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-article-limit", type=int, default=3)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--user-agent", default="")
    parser.add_argument("--contact", default=os.environ.get("SHISHA_SCRAPER_CONTACT", ""))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logger = configure_logger(args.log_level)
    root = Path(__file__).resolve().parents[1]
    output_dir = Path(args.output_dir)
    raw_dir = Path(args.raw_dir)
    report_dir = Path(args.report_dir)

    user_agent = args.user_agent or default_user_agent(args.contact)
    session = requests.Session()
    robots = check_robots_txt(args.start_url, user_agent, session=session, timeout=args.timeout)
    logger.info("robots.txt: %s (status=%s, allowed=%s)", robots.robots_txt_url, robots.status_code, robots.allowed)
    if not robots.allowed:
        raise RuntimeError(
            "robots.txt が開始URLを許可していません。許可取得済みでも、自動実行は停止します。"
        )

    raw_store = None if args.dry_run else RawHtmlStore(raw_dir)
    client = SafeHttpClient(
        user_agent=user_agent,
        allowed_domain=DEFAULT_ALLOWED_DOMAIN,
        delay=args.delay,
        jitter=args.jitter,
        timeout=args.timeout,
        retries=args.retries,
        backoff_factor=args.backoff_factor,
        max_response_bytes=args.max_response_bytes,
        session=session,
        logger=logger,
        raw_store=raw_store,
        resume=args.resume and not args.dry_run,
    )
    brand_candidates = read_brand_candidates(Path(args.master_csv))
    scrape_result = scrape_category_and_articles(
        start_url=args.start_url,
        client=client,
        max_pages=args.max_pages,
        dry_run=args.dry_run,
        max_article_fetches=args.dry_run_article_limit,
        brand_candidates=brand_candidates,
        logger=logger,
    )

    if args.dry_run:
        print(dry_run_output(scrape_result))
        print(f"- source_site: {SOURCE_SITE}")
        print(f"- source_type: {SOURCE_TYPE}")
        print(f"- robots_note: {robots.note}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    article_csv = output_dir / "shisha_lagos_articles.csv"
    paragraph_csv = output_dir / "shisha_lagos_paragraphs.csv"
    tables_csv = output_dir / "shisha_lagos_tables.csv"
    list_items_csv = output_dir / "shisha_lagos_list_items.csv"
    recommended_csv = output_dir / "shisha_lagos_recommended_mix_sections.csv"
    duplicates_csv = output_dir / "shisha_lagos_duplicates.csv"

    write_csv(scrape_result["articles_df"], article_csv)
    write_csv(scrape_result["paragraph_df"], paragraph_csv)
    write_csv(scrape_result["tables_df"], tables_csv)
    write_csv(scrape_result["list_items_df"], list_items_csv)
    write_csv(scrape_result["recommended_df"], recommended_csv)
    write_csv(scrape_result["duplicates_df"], duplicates_csv)
    summary_csv, report_md = write_quality_outputs(report_dir, scrape_result["summary"], scrape_result["fetched_records"])
    metadata = build_metadata(
        start_url=args.start_url,
        git_commit=git_commit_hash(root),
        delay=args.delay,
        user_agent=user_agent,
        max_pages=args.max_pages,
        summary=scrape_result["summary"],
        robots=robots,
        output_files=[
            str(article_csv),
            str(paragraph_csv),
            str(tables_csv),
            str(list_items_csv),
            str(recommended_csv),
            str(duplicates_csv),
            str(summary_csv),
            str(report_md),
        ],
    )
    metadata_path = report_dir / "shisha_lagos_scraping_metadata.json"
    write_metadata(metadata_path, metadata)

    print("shisha lagos scraping completed")
    print(f"- category_page_count: {scrape_result['summary']['category_page_count']}")
    print(f"- article_url_count: {scrape_result['summary']['article_url_count']}")
    print(f"- successful_article_count: {scrape_result['summary']['successful_article_count']}")
    print(f"- failed_article_count: {scrape_result['summary']['failed_article_count']}")
    print(f"- duplicate_article_count: {scrape_result['summary']['duplicate_article_count']}")
    print(f"- output_articles: {article_csv}")
    print(f"- output_paragraphs: {paragraph_csv}")
    print(f"- output_tables: {tables_csv}")
    print(f"- output_list_items: {list_items_csv}")
    print(f"- output_recommended_mix: {recommended_csv}")
    print(f"- output_duplicates: {duplicates_csv}")
    print(f"- output_summary: {summary_csv}")
    print(f"- output_report: {report_md}")
    print(f"- output_metadata: {metadata_path}")


if __name__ == "__main__":
    main()
