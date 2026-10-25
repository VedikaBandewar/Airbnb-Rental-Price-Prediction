"""Web scraping utilities: HTTP requests, robots.txt check, HTML parsing and pagination.

IMPORTANT: Airbnb's terms of service prohibit automated scraping of airbnb.com. Use this module only on sites
you are allowed to scrape (check robots.txt + terms), on public datasets, or on the built-in offline demo
pages. For real Airbnb data download Inside Airbnb CSVs (http://insideairbnb.com/get-the-data) and upload them.
"""
import re
import time
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

USER_AGENT = "SI-Practical-Educational-Scraper/1.0"
DEFAULT_SELECTORS = {  # (html tag, css class) for each field
    "card": ("div", "listing-card"), "title": ("h3", "listing-title"), "price": ("span", "price"),
    "location": ("span", "location"), "room_type": ("span", "room-type"),
    "reviews": ("span", "reviews"), "availability": ("span", "availability"),
}


def check_robots(url: str, user_agent: str = USER_AGENT, timeout: int = 5):
    """Return (allowed: bool | None, message). None means robots.txt could not be read."""
    parts = urlparse(url)
    robots_url = f"{parts.scheme}://{parts.netloc}/robots.txt"
    try:
        resp = requests.get(robots_url, timeout=timeout, headers={"User-Agent": user_agent})
        if resp.status_code == 404:
            return True, "No robots.txt found (no explicit restrictions) — still check the site's terms."
        if resp.status_code != 200:
            return None, f"robots.txt returned HTTP {resp.status_code}."
        rp = RobotFileParser()
        rp.parse(resp.text.splitlines())
        ok = rp.can_fetch(user_agent, url)
        return ok, "Allowed by robots.txt." if ok else "Disallowed by robots.txt — do not scrape this URL."
    except requests.RequestException as e:
        return None, f"Could not read robots.txt: {e}"


def fetch_page(url: str, timeout: int = 10, retries: int = 2, backoff: float = 1.0, headers: dict | None = None) -> dict:
    """HTTP GET with retries. Never raises; inspect result['ok'], ['status_code'] and ['error']."""
    hdr = {"User-Agent": USER_AGENT, **(headers or {})}
    last_err = None
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, headers=hdr, timeout=timeout)
            res = {"ok": r.ok, "url": r.url, "status_code": r.status_code, "reason": r.reason,
                   "elapsed_s": r.elapsed.total_seconds(), "content_type": r.headers.get("Content-Type", ""),
                   "encoding": r.encoding, "bytes": len(r.content), "text": r.text, "headers": dict(r.headers), "error": None}
            if r.status_code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(backoff * (2 ** attempt))
                continue
            if not r.ok:
                res["error"] = f"HTTP {r.status_code} {r.reason}"
            return res
        except requests.RequestException as e:
            last_err = str(e)
            if attempt < retries:
                time.sleep(backoff * (2 ** attempt))
    return {"ok": False, "url": url, "status_code": None, "reason": None, "elapsed_s": None, "content_type": "",
            "encoding": None, "bytes": 0, "text": "", "headers": {}, "error": last_err}


def _text(node):
    return node.get_text(strip=True) if node is not None else None


def parse_listings(html: str, selectors: dict | None = None, base_url: str = "") -> list:
    """Parse listing cards with find()/find_all()/get_text() and attribute extraction. Missing fields -> None."""
    sel = {**DEFAULT_SELECTORS, **(selectors or {})}
    soup = BeautifulSoup(html, "lxml") if _has_lxml() else BeautifulSoup(html, "html.parser")
    tag, cls = sel["card"]
    rows = []
    for card in soup.find_all(tag, class_=cls):
        row = {"listing_id": card.get("data-id")}
        for field in ("title", "price", "location", "room_type", "reviews", "availability"):
            t, c = sel[field]
            row[field] = _text(card.find(t, class_=c))
        a = card.find("a", href=True)                                   # attribute extraction
        row["url"] = urljoin(base_url, a["href"]) if a else None
        rows.append(row)
    return rows


def _has_lxml() -> bool:
    try:
        import lxml  # noqa: F401
        return True
    except ImportError:
        return False


def _num(text, as_int=False):
    if text is None:
        return np.nan
    m = re.search(r"\d[\d,]*\.?\d*", str(text))
    if not m:
        return np.nan
    v = float(m.group().replace(",", ""))
    return int(v) if as_int else v


def clean_scraped(df: pd.DataFrame) -> pd.DataFrame:
    """Convert scraped text to analysis-ready columns named like the main dataset."""
    out = pd.DataFrame()
    out["id"] = df.get("listing_id")
    out["name"] = df.get("title")
    out["neighbourhood"] = df.get("location")
    out["room_type"] = df.get("room_type")
    out["number_of_reviews"] = df["reviews"].map(_num) if "reviews" in df else np.nan
    out["availability_365"] = df["availability"].map(_num) if "availability" in df else np.nan
    out["price"] = df["price"].map(_num) if "price" in df else np.nan
    out["url"] = df.get("url")
    return out.drop_duplicates().reset_index(drop=True)


def scrape_pages(url_template: str, start: int = 1, end: int = 1, delay: float = 1.0, selectors: dict | None = None,
                 respect_robots: bool = True, fetcher=fetch_page, **fetch_kwargs):
    """Scrape url_template.format(page=n) for n in [start, end]. Returns (DataFrame, log list).
    Stops early on an empty page; records (rather than raises) every failure."""
    log, frames = [], []
    if respect_robots:
        ok, msg = check_robots(url_template.format(page=start))
        log.append({"page": "-", "status": "robots", "listings": 0, "note": msg})
        if ok is False:
            return pd.DataFrame(), log
    for n in range(start, end + 1):
        url = url_template.format(page=n)
        res = fetcher(url, **fetch_kwargs)
        if not res["ok"]:
            log.append({"page": n, "status": res["status_code"], "listings": 0, "note": res["error"]})
            if res["status_code"] == 404:
                break
            continue
        rows = parse_listings(res["text"], selectors, base_url=url)
        log.append({"page": n, "status": res["status_code"], "listings": len(rows),
                    "note": "ok" if rows else "no listings found — stopping"})
        if not rows:
            break
        frames.append(pd.DataFrame(rows))
        if n < end:
            time.sleep(delay)
    if not frames:
        return pd.DataFrame(), log
    return clean_scraped(pd.concat(frames, ignore_index=True)), log


# ------------------------- offline demo (no network needed) -------------------------
def build_demo_page(df: pd.DataFrame, page: int = 1, per_page: int = 20, seed: int = 0) -> str:
    """Render listings as HTML. Some fields are randomly omitted to mimic missing information."""
    rng = np.random.default_rng(seed + page)
    chunk = df.iloc[(page - 1) * per_page: page * per_page]

    def span(cls, text, p_missing=0.05):
        return "" if (text is None or rng.random() < p_missing) else '<span class="%s">%s</span>' % (cls, text)

    cards = []
    for _, r in chunk.iterrows():
        rid = r.get("id", "")
        reviews = r.get("number_of_reviews")
        avail = r.get("availability_365")
        parts = [
            '<div class="listing-card" data-id="%s">' % rid,
            '<h3 class="listing-title">%s</h3>' % r.get("name", ""),
            span("price", "$%s / night" % format(r["price"], ",.0f"), 0.02),
            span("location", r.get("neighbourhood", "")),
            span("room-type", r.get("room_type", "")),
            span("reviews", "%d reviews" % reviews if pd.notna(reviews) else None),
            span("availability", "%d days available" % avail if pd.notna(avail) else None),
            '<a href="/rooms/%s">View</a></div>' % rid,
        ]
        cards.append("".join(parts))
    return "<html><body><h1>Demo listings - page %d</h1>%s</body></html>" % (page, "".join(cards))


def demo_scrape(df: pd.DataFrame, pages: int = 3, per_page: int = 20):
    """Run the full parse -> clean pipeline on generated pages (offline). Returns (DataFrame, log)."""
    base = df[df["price"].notna() & (df["price"] > 0)].drop_duplicates().reset_index(drop=True)
    frames, log = [], []
    for p in range(1, pages + 1):
        html = build_demo_page(base, p, per_page)
        rows = parse_listings(html, base_url="https://demo.local")
        log.append({"page": p, "status": 200, "listings": len(rows), "note": "ok (offline demo)" if rows else "empty page"})
        if not rows:
            break
        frames.append(pd.DataFrame(rows))
    return (clean_scraped(pd.concat(frames, ignore_index=True)) if frames else pd.DataFrame()), log
