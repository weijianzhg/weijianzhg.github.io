#!/usr/bin/env python3
"""Check the generated homepage, feed entries, assets and preserved URLs before publishing."""
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "_site"
FEATURED = {
    "https://notesfromzero.substack.com/p/building-software-for-agents",
    "https://notesfromzero.substack.com/p/auto-research-for-classic-machine",
}


class Homepage(HTMLParser):
    def __init__(self):
        super().__init__()
        self.h1 = 0
        self.scripts = 0
        self.ids = set()
        self.links = []
        self.assets = []
        self.featured = []
        self.latest = []
        self.latest_titles = []
        self.in_title = False
        self.text = []
        self.canonical = None

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        self.h1 += tag == "h1"
        self.scripts += tag == "script"
        if "id" in attrs:
            self.ids.add(attrs["id"])
        if tag == "a":
            self.links.append(attrs.get("href", ""))
            if attrs.get("class") == "essay":
                self.featured.append(attrs["href"])
            if attrs.get("class") == "latest-post":
                self.latest.append(attrs["href"])
        if tag == "h4":
            self.in_title = True
            self.latest_titles.append("")
        if tag == "link":
            if attrs.get("rel") == "canonical":
                self.canonical = attrs["href"]
            if attrs.get("href", "").startswith("/"):
                self.assets.append(attrs["href"])
        if tag == "meta" and attrs.get("name") == "robots":
            assert "noindex" not in attrs.get("content", "")

    def handle_endtag(self, tag):
        if tag == "h4":
            self.in_title = False

    def handle_data(self, data):
        self.text.append(data)
        if self.in_title:
            self.latest_titles[-1] += data


def main():
    html = (BUILD / "index.html").read_text()
    page = Homepage()
    page.feed(html)
    assert page.h1 == 1 and page.scripts == 0, "Homepage must be static HTML with one main heading"
    assert page.canonical == "https://weijian.ai/"
    assert set(page.featured) == FEATURED, "Keep both highlighted essays"
    assert len(page.latest) == 3 and len(set(page.latest)) == 3
    assert not set(page.latest) & FEATURED, "Latest writing must not repeat the highlights"
    text = " ".join(page.text)
    assert "I build AI systems" in text and "rely on." in text
    assert "↗" not in text, "Use SVG link arrows so phones cannot render them as emoji"
    assert "wz." not in text and "Three directions" not in text
    assert "/_next/" not in html and "{{" not in html and "{%" not in html
    for href in page.links:
        if href.startswith("#"):
            assert href[1:] in page.ids, f"Broken section link: {href}"
    for href in page.assets:
        assert (BUILD / urlsplit(href).path.lstrip("/")).is_file(), f"Missing asset: {href}"
    css = (BUILD / "assets/editorial/editorial.css").read_text()
    assert ".editorial-hero" in css and ".latest-post" in css
    for href in re.findall(r"url\(['\"]?([^)'\"]+)", css):
        if href.startswith("/"):
            assert (BUILD / href.lstrip("/")).is_file(), f"Missing font: {href}"
    entries = ET.parse(ROOT / "_cache/notesfromzero.substack.com_feed.xml").findall("./channel/item")
    entries.sort(key=lambda item: parsedate_to_datetime(item.findtext("pubDate")), reverse=True)
    expected = []
    seen = set()
    for item in entries:
        url = item.findtext("link")
        if url not in FEATURED and url not in seen:
            expected.append((url, item.findtext("title")))
            seen.add(url)
    assert list(zip(page.latest, page.latest_titles)) == expected[:3], "Latest list must match the refreshed feed"
    for path in ["building/index.html", "writing/index.html", "lattice-graph.html"]:
        assert (BUILD / path).is_file(), f"Preserve existing URL: {path}"
    assert (BUILD / "CNAME").read_text().strip() == "weijian.ai"
    print("Static homepage verified: current feed, featured essays, fonts, links, archive URLs and domain.")


if __name__ == "__main__":
    main()
