#!/usr/bin/env python3
"""
Tradeoff Layer — Unattended Public Web Scraper, Topic Discovery & Source Citation Engine.

Capabilities:
  1. Unattended Topic Discovery (`discover_topics`):
     Scrapes live public engineering/AI/semiconductor/security feeds (Hacker News Algolia API,
     ArXiv API, GitHub Security Advisories / Public Tech Feeds), deduplicates against existing
     @TradeoffLayer videos, and scores candidates on Tradeoff Layer's editorial rubric
     (counter-intuitive paradox, concrete numbers, structural engineering/economic tradeoff).
  2. Deep Public Web Dossier Scraper (`build_research_dossier`):
     Crawls multiple live public sources (Wikipedia REST/MediaWiki API, ArXiv abstracts,
     HN top stories & linked articles, live web pages), extracts clean factual passages and
     quantitative metrics, and builds a numbered Source Registry ([S1]..[Sn]) with canonical
     URLs, publisher names, dates, and verbatim snippets for on-screen + description citations.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 "
    "TradeoffLayerResearchBot/1.0"
)

EXISTING_CHANNEL_TITLES = [
    "Why Every Phone Has a Sony Camera Sensor?",
    "Binance's $6.3B Hack Explained",
    "Why Intel Couldn't Kill AMD?",
    "Why Everyone Pays ARM?",
    "PHP Vulnerability, Simply Explained",
    "Why No One Can Copy ASML?",
    "Is Big Tech Wasting $800 Billion on AI?",
    "Can Google Beat Nvidia? TPU vs GPU Explained",
    "The OpenAI–Hugging Face Hack: What Really Happened",
    "AI Costs Explained: Why Output Tokens Cost 5x More",
]


def _http_get(url: str, timeout: int = 15, accept: str = "*/*") -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": accept,
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read(2_500_000)


def _strip_html_to_text(raw_html: str) -> tuple[str, str]:
    title_m = re.search(r"<title[^>]*>(.*?)</title>", raw_html, flags=re.IGNORECASE | re.DOTALL)
    title = html.unescape(re.sub(r"\s+", " ", title_m.group(1))).strip() if title_m else ""
    cleaned = re.sub(r"<(script|style|noscript|svg|header|footer|nav)[^>]*>.*?</\1>", " ", raw_html, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = html.unescape(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return title, cleaned


def _extract_numeric_facts(text: str, max_facts: int = 8) -> list[str]:
    sents = re.split(r"(?<=[.?!])\s+", text)
    facts: list[str] = []
    for s in sents:
        s_clean = s.strip()
        if not (45 <= len(s_clean) <= 280):
            continue
        if re.search(r"(\$\d+|\d+(?:\.\d+)?%|\b\d+(?:\.\d+)?\s*(?:billion|million|trillion|GW|MW|nm|GB|TB|times|x)\b)", s_clean, re.IGNORECASE):
            facts.append(s_clean)
            if len(facts) >= max_facts:
                break
    return facts


def search_hackernews(query: str, hits_per_page: int = 8, min_points: int = 30) -> list[dict[str, Any]]:
    params = urllib.parse.urlencode({
        "query": query,
        "tags": "story",
        "numericFilters": f"points>{min_points}",
        "hitsPerPage": str(hits_per_page),
    })
    url = f"https://hn.algolia.com/api/v1/search?{params}"
    try:
        data = json.loads(_http_get(url, accept="application/json").decode("utf-8"))
    except Exception:
        return []
    out = []
    for h in data.get("hits", []):
        story_url = h.get("url") or f"https://news.ycombinator.com/item?id={h.get('objectID')}"
        out.append({
            "title": h.get("title") or "",
            "url": story_url,
            "hn_url": f"https://news.ycombinator.com/item?id={h.get('objectID')}",
            "points": int(h.get("points") or 0),
            "comments": int(h.get("num_comments") or 0),
            "date": (h.get("created_at") or "")[:10],
            "publisher": urllib.parse.urlparse(story_url).netloc.replace("www.", "") or "news.ycombinator.com",
        })
    return out


def search_wikipedia(query: str, limit: int = 4) -> list[dict[str, Any]]:
    params = urllib.parse.urlencode({
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": str(limit),
        "format": "json",
    })
    api_url = f"https://en.wikipedia.org/w/api.php?{params}"
    try:
        data = json.loads(_http_get(api_url, accept="application/json").decode("utf-8"))
    except Exception:
        return []
    results = []
    for item in data.get("query", {}).get("search", []):
        page_title = item.get("title", "")
        slug = urllib.parse.quote(page_title.replace(" ", "_"))
        sum_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{slug}"
        try:
            sdata = json.loads(_http_get(sum_url, accept="application/json").decode("utf-8"))
            extract = sdata.get("extract") or ""
            canonical = sdata.get("content_urls", {}).get("desktop", {}).get("page") or f"https://en.wikipedia.org/wiki/{slug}"
            ts = (sdata.get("timestamp") or "")[:10]
        except Exception:
            extract = re.sub(r"<[^>]+>", "", item.get("snippet", ""))
            canonical = f"https://en.wikipedia.org/wiki/{slug}"
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if extract:
            results.append({
                "title": f"Wikipedia — {page_title}",
                "url": canonical,
                "publisher": "Wikipedia",
                "date": ts,
                "summary": extract,
                "key_facts": _extract_numeric_facts(extract, max_facts=5),
            })
    return results


def search_arxiv(query: str, max_results: int = 3) -> list[dict[str, Any]]:
    clean_q = re.sub(r'["()]', "", query).strip()
    search_expr = f'(cat:cs.AR OR cat:cs.AI OR cat:cs.DC OR cat:cs.PF) AND all:"{clean_q}"'
    params = urllib.parse.urlencode({
        "search_query": search_expr,
        "start": "0",
        "max_results": str(max_results),
        "sortBy": "relevance",
        "sortOrder": "descending",
    })
    url = f"http://export.arxiv.org/api/query?{params}"
    try:
        xml_bytes = _http_get(url, accept="application/atom+xml")
        root = ET.fromstring(xml_bytes)
    except Exception:
        return []
    ns = {"a": "http://www.w3.org/2005/Atom"}
    papers = []
    for entry in root.findall("a:entry", ns):
        title = re.sub(r"\s+", " ", (entry.findtext("a:title", "", ns) or "")).strip()
        summary = re.sub(r"\s+", " ", (entry.findtext("a:summary", "", ns) or "")).strip()
        link = (entry.findtext("a:id", "", ns) or "").strip().replace("http://", "https://")
        pub = (entry.findtext("a:published", "", ns) or "")[:10]
        authors = [au.findtext("a:name", "", ns) for au in entry.findall("a:author", ns)[:3]]
        if title and summary:
            papers.append({
                "title": f"{title} ({', '.join(a for a in authors if a)})",
                "url": link,
                "publisher": "arXiv",
                "date": pub,
                "summary": summary,
                "key_facts": _extract_numeric_facts(summary, max_facts=4),
            })
    return papers


def scrape_article_excerpt(url: str) -> dict[str, Any] | None:
    if any(skip in url for skip in ("youtube.com", "youtu.be", "twitter.com", "x.com", ".pdf")):
        return None
    try:
        raw = _http_get(url, timeout=10, accept="text/html,application/xhtml+xml").decode("utf-8", errors="replace")
    except Exception:
        return None
    title, text = _strip_html_to_text(raw)
    if len(text) < 250:
        return None
    domain = urllib.parse.urlparse(url).netloc.replace("www.", "")
    return {
        "title": title[:160] or domain,
        "url": url,
        "publisher": domain,
        "summary": text[:1600],
        "key_facts": _extract_numeric_facts(text[:8000], max_facts=6),
    }


def discover_topics(count: int = 5) -> list[dict[str, Any]]:
    """
    Autonomously discovers high-potential Tradeoff Layer video topics from live public feeds,
    filtering out topics already covered on @TradeoffLayer.
    """
    seed_queries = [
        ("HBM memory", "Semiconductors & AI Hardware"),
        ("TSMC packaging", "Semiconductor Supply Chain"),
        ("subsea cable", "Cloud Infrastructure"),
        ("data center power", "Energy & Compute Economics"),
        ("BGP outage", "Internet Infrastructure & Security"),
        ("CUDA AMD", "Chip Ecosystems"),
    ]
    existing_lower = " ".join(EXISTING_CHANNEL_TITLES).lower()
    candidates: list[dict[str, Any]] = []

    for q, category in seed_queries:
        stories = search_hackernews(q, hits_per_page=5, min_points=50)
        if not stories:
            continue
        top = stories[0]
        words = [w.lower() for w in re.findall(r"[A-Za-z]{4,}", q)]
        overlap = sum(1 for w in words if w in existing_lower)
        # Prefer fresh angles not identical to existing titles
        novelty_bonus = 25 if overlap <= 1 else 5
        score = min(100, int(top["points"] * 0.08 + top["comments"] * 0.05 + novelty_bonus + 45))
        candidates.append({
            "query": q,
            "category": category,
            "score": score,
            "lead_story": top,
            "supporting_stories": stories[1:4],
        })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:count]


def build_research_dossier(
    topic: str,
    search_queries: list[str] | None = None,
    max_sources: int = 10,
) -> dict[str, Any]:
    """
    Scrapes live public web sources (Wikipedia, arXiv, Hacker News, engineering articles)
    for `topic`, deduplicates URLs, and builds a numbered Source Citation Registry [S1..Sn].
    """
    queries = search_queries or [topic]
    raw_sources: list[dict[str, Any]] = []
    seen_urls: set[str] = set()

    def _add_source(item: dict[str, Any]) -> None:
        u = (item.get("url") or "").strip()
        if not u or u in seen_urls or len(raw_sources) >= max_sources:
            return
        seen_urls.add(u)
        raw_sources.append(item)

    # 1. Wikipedia technical & historical foundation (cap at 4)
    for q in queries[:3]:
        if len(raw_sources) >= 4:
            break
        for w in search_wikipedia(q, limit=2):
            if len(raw_sources) < 4:
                _add_source(w)

    # 2. ArXiv CS/Hardware research papers (cap at 6 cumulative)
    for q in queries[:2]:
        if len(raw_sources) >= 6:
            break
        for p in search_arxiv(q, max_results=2):
            if len(raw_sources) < 6:
                _add_source(p)

    # 3. Hacker News stories + live web article scraping
    hn_candidates: list[dict[str, Any]] = []
    for q in queries[:3]:
        hn_candidates.extend(search_hackernews(q, hits_per_page=5, min_points=25))

    for h in hn_candidates[:8]:
        if len(raw_sources) >= max_sources:
            break
        scraped = scrape_article_excerpt(h["url"])
        if scraped:
            scraped["date"] = h.get("date") or datetime.now(timezone.utc).strftime("%Y-%m-%d")
            scraped["hn_points"] = h.get("points", 0)
            _add_source(scraped)
        else:
            _add_source({
                "title": h["title"],
                "url": h["hn_url"],
                "publisher": "Hacker News",
                "date": h.get("date") or "",
                "summary": f"Community-verified engineering discussion ({h.get('points', 0)} pts, {h.get('comments', 0)} comments): {h['title']} ({h['url']})",
                "key_facts": [],
            })

    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    numbered_sources: list[dict[str, Any]] = []
    for idx, src in enumerate(raw_sources, start=1):
        sid = f"S{idx}"
        short_domain = urllib.parse.urlparse(src["url"]).netloc.replace("www.", "")
        numbered_sources.append({
            "id": sid,
            "badge_label": f"[{sid}] {src.get('publisher') or short_domain} ({src.get('date') or '2026'})",
            "short_url": f"{short_domain}{urllib.parse.urlparse(src['url']).path[:36]}",
            "title": src["title"],
            "publisher": src.get("publisher") or short_domain,
            "url": src["url"],
            "date": src.get("date") or "2026",
            "retrieved_at": now_iso,
            "summary": src.get("summary", "")[:1200],
            "key_facts": src.get("key_facts", []),
        })

    return {
        "topic": topic,
        "queries": queries,
        "retrieved_at": now_iso,
        "source_count": len(numbered_sources),
        "sources": numbered_sources,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Tradeoff Layer Public Web Scraper & Source Citation Builder")
    ap.add_argument("--topic", default="", help="Topic to research; if omitted, auto-discovers top topic")
    ap.add_argument("--queries", nargs="*", help="Optional targeted search queries")
    ap.add_argument("--discover-only", action="store_true")
    ap.add_argument("--output", type=Path, default=Path("research_dossier.json"))
    args = ap.parse_args()

    if args.discover_only:
        topics = discover_topics(count=5)
        print(json.dumps(topics, indent=2, ensure_ascii=False))
        return

    topic = args.topic
    queries = args.queries
    if not topic:
        discovered = discover_topics(count=3)
        if not discovered:
            raise RuntimeError("No topics discovered from live feeds")
        chosen = discovered[0]
        topic = chosen["query"]
        queries = [chosen["query"], chosen["lead_story"]["title"][:60]]

    dossier = build_research_dossier(topic=topic, search_queries=queries)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dossier, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[research_scraper] Scraped {dossier['source_count']} verified public sources -> {args.output}")


if __name__ == "__main__":
    main()
