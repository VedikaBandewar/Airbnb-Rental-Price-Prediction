import pandas as pd
import streamlit as st
from utils import ui
from utils.data_io import validate_dataset
from utils.scraping import (DEFAULT_SELECTORS, check_robots, demo_scrape, fetch_page, parse_listings, scrape_pages)

ui.page_header("Web Scraping", "Requests + BeautifulSoup: fetch, inspect, parse, paginate and export listing data.")
ui.note("⚠️ <b>Legal note:</b> Airbnb's terms prohibit automated scraping of airbnb.com. Use the offline demo, a site you "
        "are permitted to scrape, or download a real dataset from <a href='http://insideairbnb.com/get-the-data' "
        "target='_blank'>Inside Airbnb</a> and upload it from the sidebar.")
tab_demo, tab_live, tab_code = st.tabs(["Offline demo (recommended)", "Live URL (permitted sites only)", "How it works"])

with tab_demo:
    st.write("Generates listing pages as HTML from the bundled dataset (some fields deliberately missing), then runs the "
             "**same parser and cleaner** used for live pages.")
    c1, c2 = st.columns(2)
    pages = c1.slider("Pages to scrape", 1, 10, 3)
    per = c2.slider("Listings per page", 5, 50, 20)
    if st.button("Run demo scrape", type="primary"):
        df, log = demo_scrape(ui.get_raw(), pages, per)
        st.session_state["demo_scrape"] = (df, log)
    if "demo_scrape" in st.session_state:
        df, log = st.session_state["demo_scrape"]
        ui.show_df(pd.DataFrame(log), hide_index=True)
        st.success(f"Scraped {len(df)} listings. Missing values per column:")
        ui.show_df(df.isna().sum().rename("missing").to_frame().T, hide_index=True)
        ui.show_df(df)
        c1, c2 = st.columns(2)
        with c1:
            ui.download_csv(df, "⬇️ Export scraped data (CSV)", "scraped_listings.csv", key="dl_demo")
        with c2:
            if st.button("Use this scraped data as the active dataset", key="use_demo"):
                ok, msgs = validate_dataset(df, min_rows=50)
                if ok:
                    st.session_state["scraped_df"] = df
                    st.session_state["_pending_source"] = ui.SOURCES[2]
                    st.rerun()
                else:
                    st.error(" ".join(msgs) + " Increase pages/listings (≥ 50 rows).")

with tab_live:
    url_t = st.text_input("URL template (use {page} for the page number)", placeholder="https://example.com/listings?page={page}")
    c1, c2, c3 = st.columns(3)
    start = c1.number_input("First page", 1, 1000, 1)
    end = c2.number_input("Last page", 1, 1000, 1)
    delay = c3.number_input("Delay between requests (s)", 0.5, 30.0, 1.0)
    st.caption("CSS selectors (tag, class) for each field — adjust them to match the target site's HTML:")
    sel = {}
    cols = st.columns(4)
    for i, (k, (tag, cls)) in enumerate(DEFAULT_SELECTORS.items()):
        with cols[i % 4]:
            t = st.text_input(f"{k}: tag", tag, key=f"t_{k}")
            c = st.text_input(f"{k}: class", cls, key=f"c_{k}")
            sel[k] = (t, c)
    b1, b2 = st.columns(2)
    if b1.button("1) Inspect first page (GET + status code)"):
        if not url_t.strip() or not url_t.startswith(("http://", "https://")):
            st.error("Enter a URL starting with http:// or https://")
        else:
            url = url_t.format(page=int(start))
            ok, msg = check_robots(url)
            (st.success if ok else st.warning)(f"robots.txt: {msg}")
            r = fetch_page(url, retries=1)
            if r["ok"]:
                st.success(f"HTTP {r['status_code']} {r['reason']} — {r['bytes']:,} bytes in {r['elapsed_s']:.2f}s ({r['content_type']})")
                st.code(r["text"][:1500], language="html")
                rows = parse_listings(r["text"], sel, base_url=url)
                st.info(f"Parser found {len(rows)} listing cards with the current selectors.")
                if rows:
                    ui.show_df(pd.DataFrame(rows).head(10))
            else:
                st.error(f"Request failed: {r['error']}")
    if b2.button("2) Scrape page range", type="primary"):
        if not url_t.strip() or "{page}" not in url_t or not url_t.startswith(("http://", "https://")):
            st.error("The URL must start with http(s):// and contain {page}.")
        elif end < start:
            st.error("Last page must be ≥ first page.")
        else:
            with st.spinner("Scraping politely..."):
                df, log = scrape_pages(url_t, int(start), int(end), delay, sel)
            ui.show_df(pd.DataFrame(log), hide_index=True)
            if df.empty:
                st.warning("No listings collected (blocked by robots.txt, request errors, or selectors do not match).")
            else:
                st.session_state["live_scrape"] = df
    if "live_scrape" in st.session_state:
        df = st.session_state["live_scrape"]
        ui.show_df(df)
        ui.download_csv(df, "⬇️ Export scraped data (CSV)", "scraped_live.csv", key="dl_live")
        if st.button("Use as active dataset", key="use_live"):
            ok, msgs = validate_dataset(df)
            if ok:
                st.session_state["scraped_df"] = df
                st.session_state["_pending_source"] = ui.SOURCES[2]
                st.rerun()
            else:
                st.error(" ".join(msgs))

with tab_code:
    st.markdown("""
**Pipeline:** `requests.get()` → check `status_code` → `BeautifulSoup(html)` → `find_all()` listing cards →
`find()` + `get_text()` for each field → `a["href"]` attribute extraction → DataFrame → clean text to numbers
(`"$1,200 / night"` → `1200`) → CSV export.

**Robustness built in:** timeouts, retries with exponential back-off, HTTP error codes recorded (not raised),
missing fields become `NaN`, pagination stops on an empty/404 page, robots.txt is checked, and a delay is applied
between requests.
""")
