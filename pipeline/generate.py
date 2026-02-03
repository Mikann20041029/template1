# pipeline/generate.py
# Lite edition: generates ONE article from RSS and builds a static site (GitHub Pages friendly).
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Tuple
from slugify import slugify
import json
import random
import re
import html as _html
import urllib.request
from urllib.parse import urlparse

from pipeline.util import (
    ROOT,
    read_text,
    write_text,
    read_json,
    write_json,
    normalize_url,
    simple_tokens,
    jaccard,
    sanitize_llm_html,
)
from pipeline.deepseek import DeepSeekClient
from pipeline.reddit import fetch_rss_entries
from pipeline.render import env_for, render_to_file, write_asset

# -----------------------------
# Paths (repo-root config for product-template simplicity)
# -----------------------------
CONFIG_PATH = ROOT / "config.json"
PROCESSED_PATH = ROOT / "processed_urls.txt"
ARTICLES_PATH = ROOT / "data" / "articles.json"
LAST_RUN_PATH = ROOT / "data" / "last_run.json"

TEMPLATES_DIR = ROOT / "pipeline" / "templates"
STATIC_DIR = ROOT / "pipeline" / "static"

# Lite: run once flag (best-effort)
LITE_RAN_FLAG = ROOT / ".lite_ran"


# -----------------------------
# Helpers
# -----------------------------
def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise RuntimeError(f"config.json not found: {CONFIG_PATH}")
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def get_site_dir(cfg: dict) -> Path:
    site_dir_str = (cfg.get("site", {}).get("site_dir") or "docs").strip()
    return ROOT / site_dir_str.lstrip("./")


def load_processed() -> set[str]:
    s = read_text(PROCESSED_PATH)
    lines = [normalize_url(x) for x in s.splitlines() if x.strip()]
    return set(lines)


def append_processed(url: str) -> None:
    url = normalize_url(url)
    existing = load_processed()
    if url in existing:
        return
    current = read_text(PROCESSED_PATH).rstrip()
    if current.strip():
        current += "\n"
    current += url + "\n"
    write_text(PROCESSED_PATH, current)


def is_blocked(title: str, blocked_kw: list[str]) -> bool:
    t = (title or "").lower()
    for kw in blocked_kw:
        if kw.lower() in t:
            return True
    return False


def compute_rankings(articles: list[dict]) -> list[dict]:
    return sorted(articles, key=lambda a: a.get("published_ts", ""), reverse=True)


def related_articles(current: dict, articles: list[dict], k: int = 6) -> list[dict]:
    cur_tok = simple_tokens(current.get("title", ""))
    scored: list[tuple[float, dict]] = []
    for a in articles:
        if a.get("id") == current.get("id"):
            continue
        sim = jaccard(cur_tok, simple_tokens(a.get("title", "")))
        scored.append((sim, a))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [a for s, a in scored[:k] if s > 0.05]


def _guess_ext_from_url(u: str) -> str:
    try:
        path = urlparse(u).path.lower()
    except Exception:
        path = ""
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        if path.endswith(ext):
            return ext
    return ".jpg"


def cache_og_image(cfg: dict, site_dir: Path, base_url: str, src_url: str, article_id: str) -> str:
    src_url = (src_url or "").strip()
    if not src_url:
        return ""
    ext = _guess_ext_from_url(src_url)
    rel = f"/og/{article_id}{ext}"
    out_path = site_dir / rel.lstrip("/")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not out_path.exists():
        try:
            ua = (cfg.get("site", {}).get("user_agent") or "").strip() or "Mozilla/5.0 (compatible; AutoSiteLiteBot/1.0)"
            req = urllib.request.Request(src_url, headers={"User-Agent": ua})
            with urllib.request.urlopen(req, timeout=20) as r:
                data = r.read()
            if data:
                out_path.write_bytes(data)
        except Exception:
            return ""
    return base_url.rstrip("/") + rel


# -----------------------------
# Candidate picking
# -----------------------------
def pick_candidate(cfg: dict, processed: set[str], articles: list[dict]) -> Optional[dict]:
    safety = cfg.get("safety", {})
    feeds = cfg.get("feeds", {})

    blocked_kw = safety.get("blocked_keywords", []) or []
    rss_list = feeds.get("reddit_rss", []) or []
    if not rss_list:
        raise RuntimeError("config.feeds.reddit_rss is empty. Add at least one RSS URL.")

    prev_titles = [a.get("title", "") for a in articles]
    prev_tok = [simple_tokens(t) for t in prev_titles if t]

    candidates: list[dict] = []
    for rss in rss_list:
        for e in fetch_rss_entries(rss):
            link = normalize_url(e.get("link", ""))
            title = e.get("title", "")
            if not link or link in processed:
                continue
            if is_blocked(title, blocked_kw):
                continue

            tok = simple_tokens(title)
            too_similar = any(jaccard(tok, pt) >= 0.78 for pt in prev_tok)
            if too_similar:
                continue

            e["image_url"] = e.get("hero_image", "") or ""
            e["image_kind"] = e.get("hero_image_kind", "none") or "none"
            candidates.append(e)

    if not candidates:
        return None

    if cfg.get("generation", {}).get("pick_random", False):
        return random.choice(candidates)
    return candidates[0]


# -----------------------------
# DeepSeek generation (Lite)
# -----------------------------
def deepseek_article(cfg: dict, item: dict) -> Tuple[str, str]:
    ds = DeepSeekClient()

    gen = cfg.get("generation", {})
    model = gen.get("model", "deepseek-chat")
    temp = float(gen.get("temperature", 0.7))
    max_tokens = int(gen.get("max_tokens", 2200))

    title = item.get("title", "").strip()
    link = item.get("link", "").strip()
    summary = (item.get("summary", "") or "").strip()

    system = (
        "You are a careful writer. Write in English only. Do not fabricate facts. "
        "If something is not stated in the source, say: 'Not stated in the source.'"
    )

    user = f"""
OUTPUT RULES:
- First line MUST be: TITLE: <your best SEO-friendly title>
- Second line MUST be empty.
- From the third line, output the HTML body only.
- Allowed tags: <p>, <h2>, <ul>, <li>, <strong>, <code>, <a>
- Do NOT output <h1>.

INPUT:
Post title: {title}
Permalink: {link}
RSS summary snippet: {summary}

STRUCTURE:
1) <p><strong>[SUMMARY]</strong>: 2 lines.</p>
2) <h2>What happened</h2> (2–4 short paragraphs)
3) <h2>Why people care</h2> (2–4 short paragraphs)
4) <h2>Practical takeaways</h2> (5 bullet points)
5) <h2>Source</h2> Link to the original post
""".strip()

    out = ds.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temp,
        max_tokens=max_tokens,
    )

    out = (out or "").strip()

    m = re.match(r"(?is)^\s*TITLE:\s*(.+?)\s*\n\s*\n(.*)$", out)
    if m:
        llm_title = m.group(1).strip()
        llm_html = m.group(2).strip()
    else:
        llm_title = title
        llm_html = out

    llm_html = sanitize_llm_html(llm_html or "")
    return (llm_title, llm_html)


# -----------------------------
# Site build
# -----------------------------
def write_rss_feed(cfg: dict, site_dir: Path, articles: list[dict], limit: int = 10) -> None:
    base_url = cfg["site"]["base_url"].rstrip("/")
    site_title = cfg["site"].get("brand_name", "AutoSite Lite")
    site_desc = cfg["site"].get("description", "Daily digest")

    items = sorted(articles, key=lambda a: a.get("published_ts", ""), reverse=True)[:limit]

    def rfc822(iso: str) -> str:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S +0000")

    parts = []
    parts.append('<?xml version="1.0" encoding="UTF-8"?>')
    parts.append("<rss version='2.0' xmlns:atom='http://www.w3.org/2005/Atom'>")
    parts.append("<channel>")
    parts.append(f"<title>{_html.escape(site_title)}</title>")
    parts.append(f"<link>{_html.escape(base_url + '/')}</link>")
    parts.append(f"<description>{_html.escape(site_desc)}</description>")
    parts.append(f"<lastBuildDate>{_html.escape(rfc822(now_utc_iso()))}</lastBuildDate>")

    for a in items:
        url = f"{base_url}{a['path']}"
        title = a.get("title", "")
        pub = a.get("published_ts", now_utc_iso())
        summary = a.get("summary", "") or ""
        if not summary:
            summary = re.sub(r"\s+", " ", re.sub(r"(?is)<[^>]+>", " ", a.get("body_html", ""))).strip()[:240]

        parts.append("<item>")
        parts.append(f"<title>{_html.escape(title)}</title>")
        parts.append(f"<link>{_html.escape(url)}</link>")
        parts.append(f"<guid isPermaLink='true'>{_html.escape(url)}</guid>")
        parts.append(f"<pubDate>{_html.escape(rfc822(pub))}</pubDate>")
        parts.append(f"<description>{_html.escape(summary)}</description>")
        parts.append("</item>")

    parts.append("</channel>")
    parts.append("</rss>")

    (site_dir / "feed.xml").write_text("\n".join(parts) + "\n", encoding="utf-8")


def build_site(cfg: dict, site_dir: Path, articles: list[dict]) -> None:
    base_url = cfg["site"]["base_url"].rstrip("/")

    site_dir.mkdir(parents=True, exist_ok=True)
    (site_dir / "articles").mkdir(parents=True, exist_ok=True)
    (site_dir / "assets").mkdir(parents=True, exist_ok=True)

    write_asset(site_dir / "assets" / "style.css", STATIC_DIR / "style.css")
    if (STATIC_DIR / "fx.js").exists():
        write_asset(site_dir / "assets" / "fx.js", STATIC_DIR / "fx.js")

    robots = f"""User-agent: *
Allow: /

Sitemap: {base_url}/sitemap.xml
"""
    (site_dir / "robots.txt").write_text(robots, encoding="utf-8")

    urls = [f"{base_url}/"] + [f"{base_url}{a['path']}" for a in articles]
    sitemap_items = "\n".join([f"<url><loc>{u}</loc></url>" for u in urls])
    sitemap = f"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
{sitemap_items}
</urlset>
"""
    (site_dir / "sitemap.xml").write_text(sitemap, encoding="utf-8")

    jenv = env_for(TEMPLATES_DIR)

    ranking = compute_rankings(articles)[:10]
    new_articles = sorted(articles, key=lambda a: a.get("published_ts", ""), reverse=True)[:10]

    write_rss_feed(cfg, site_dir, articles, limit=10)

    base_ctx = {
        "site": cfg["site"],
        "ranking": ranking,
        "new_articles": new_articles,
        "ads_top": "",
        "ads_mid": "",
        "ads_bottom": "",
        "ads_rail_left": "",
        "ads_rail_right": "",
        "now_iso": now_utc_iso(),
    }

    ctx = dict(base_ctx)
    ctx.update(
        {
            "title": cfg["site"].get("brand_name", "AutoSite Lite"),
            "description": cfg["site"].get("description", "Daily digest"),
            "canonical": base_url + "/",
            "og_type": "website",
            "og_image": "",
        }
    )
    render_to_file(jenv, "index.html", ctx, site_dir / "index.html")

    static_pages = [
        ("about", "About", "<p>This is a lite demo build: one run, one generated article.</p>"),
        ("privacy", "Privacy", "<p>No accounts required. Third-party services may collect device identifiers.</p>"),
        ("terms", "Terms", "<p>Use at your own risk. No guarantees.</p>"),
        ("disclaimer", "Disclaimer", "<p>Not affiliated with any source. Trademarks belong to their owners.</p>"),
        ("contact", "Contact", f"<p>Email: <a href='mailto:{cfg['site']['contact_email']}'>{cfg['site']['contact_email']}</a></p>"),
    ]
    for slug, page_title, body in static_pages:
        ctx = dict(base_ctx)
        ctx.update(
            {
                "page_title": page_title,
                "page_body": body,
                "title": page_title,
                "description": cfg["site"].get("description", "Daily digest"),
                "canonical": f"{base_url}/{slug}.html",
                "og_type": "website",
                "og_image": "",
            }
        )
        render_to_file(jenv, "static.html", ctx, site_dir / f"{slug}.html")

    og_cfg = cfg.get("og", {})
    og_cache_enabled = bool(og_cfg.get("cache_images", True))

    for a in articles:
        rel = related_articles(a, articles, k=6)
        src = a.get("hero_image", "") or ""
        og_img = ""
        if og_cache_enabled:
            og_img = cache_og_image(cfg, site_dir, base_url, src, a.get("id", "article"))

        ctx = dict(base_ctx)
        ctx.update(
            {
                "a": a,
                "related": rel,
                "policy_block": "",
                "title": a.get("title", cfg["site"].get("brand_name", "AutoSite Lite")),
                "description": (a.get("summary", "") or cfg["site"].get("description", "Daily digest"))[:200],
                "canonical": f"{base_url}{a['path']}",
                "og_type": "article",
                "og_image": og_img,
            }
        )
        render_to_file(jenv, "article.html", ctx, site_dir / a["path"].lstrip("/"))


def write_last_run(cfg: dict, payload: dict[str, Any]) -> None:
    base_url = cfg["site"]["base_url"].rstrip("/")
    out = {"updated_utc": now_utc_iso(), "homepage_url": base_url + "/", **payload}
    write_json(LAST_RUN_PATH, out)


def main() -> None:
    if LITE_RAN_FLAG.exists():
        raise RuntimeError("Lite edition: this repository can only be executed once.")

    cfg = load_config()

    base_url = (cfg.get("site", {}).get("base_url") or "").strip()
    if not base_url:
        raise RuntimeError("config.site.base_url is missing (example: https://YOURNAME.github.io/YOURREPO)")

    site_dir = get_site_dir(cfg)

    processed = load_processed()
    articles = read_json(ARTICLES_PATH, default=[])

    cand = pick_candidate(cfg, processed, articles)
    if not cand:
        build_site(cfg, site_dir, articles)
        write_last_run(cfg, {"created": False, "article_url": "", "article_title": "", "source_url": "", "note": "No new candidate found. Site rebuilt."})
        LITE_RAN_FLAG.write_text(now_utc_iso() + "\n", encoding="utf-8")
        return

    llm_title, body_html = deepseek_article(cfg, cand)

    ts = datetime.now(timezone.utc)
    ymd = ts.strftime("%Y-%m-%d")
    slug = slugify(llm_title or cand.get("title", ""))[:80] or f"post-{int(ts.timestamp())}"
    path = f"/articles/{ymd}-{slug}.html"
    article_url = base_url.rstrip("/") + path

    entry = {
        "id": f"{ymd}-{slug}",
        "title": llm_title or cand.get("title", ""),
        "path": path,
        "published_ts": ts.isoformat(timespec="seconds"),
        "source_url": cand.get("link", ""),
        "rss": cand.get("rss", ""),
        "summary": cand.get("summary", ""),
        "body_html": body_html,
        "hero_image": cand.get("image_url", "") or "",
        "hero_image_kind": cand.get("image_kind", "none") or "none",
    }

    append_processed(cand.get("link", ""))
    articles.insert(0, entry)
    write_json(ARTICLES_PATH, articles)

    build_site(cfg, site_dir, articles)

    write_last_run(cfg, {"created": True, "article_url": article_url, "article_path": path, "article_title": cand.get("title", ""), "source_url": cand.get("link", "")})

    LITE_RAN_FLAG.write_text(now_utc_iso() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
