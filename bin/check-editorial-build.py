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
# The highlighted essays, read from the same data file the pages use.
FEATURED = set(re.findall(r"^\s+url:\s*(\S+)", (ROOT / "_data/start_here.yml").read_text(), re.M))
assert len(FEATURED) == 3, "Highlight three essays in _data/start_here.yml"
LATEST_COUNT = 5
FOLLOW = {
    "https://notesfromzero.substack.com/subscribe",
    "https://notesfromzero.substack.com",
    "https://latticeworkofmodels.substack.com",
    "https://x.com/Weijian",
    "https://www.linkedin.com/in/weijianzhg",
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
        self.in_latest = False
        self.in_title = False
        self.text = []
        self.canonical = None
        self.images = []
        self.og_image = None

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
            self.in_latest = attrs.get("class") == "latest-post"
            if self.in_latest:
                self.latest.append(attrs["href"])
        if tag == "h4" and self.in_latest:
            self.in_title = True
            self.latest_titles.append("")
        if tag == "img":
            self.images.append(attrs)
        if tag == "meta" and attrs.get("property") == "og:image":
            self.og_image = attrs["content"]
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
        if tag == "a":
            self.in_latest = False

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
    assert set(page.featured) == FEATURED and len(page.featured) == len(FEATURED), "Keep the highlighted essays"
    assert len(page.latest) == LATEST_COUNT and len(set(page.latest)) == LATEST_COUNT
    assert not set(page.latest) & FEATURED, "Latest writing must not repeat the highlights"
    text = " ".join(page.text)
    assert "I build AI for finance" in text and "Hedge Layer" in text
    assert FOLLOW <= set(page.links), "Keep the subscribe link and every place to follow"
    assert len(page.images) == 1, "The homepage shows one project"
    for image in page.images:
        assert image.get("alt"), "Images need alt text"
        sources = [image["src"]] + [part.split()[0] for part in image.get("srcset", "").split(",") if part.strip()]
        for src in sources:
            assert (BUILD / src.lstrip("/")).is_file(), f"Missing image: {src}"
    assert page.og_image and page.og_image.startswith("https://weijian.ai/"), "Share image must be an absolute URL"
    assert (BUILD / urlsplit(page.og_image).path.lstrip("/")).is_file(), "Missing share image"
    assert "↗" not in text, "Use SVG link arrows so phones cannot render them as emoji"
    assert "wz." not in text and "Three directions" not in text
    assert "/_next/" not in html and "{{" not in html and "{%" not in html
    for href in page.links:
        if href.startswith("#"):
            assert href[1:] in page.ids, f"Broken section link: {href}"
    for href in page.assets:
        assert (BUILD / urlsplit(href).path.lstrip("/")).is_file(), f"Missing asset: {href}"
    css = (BUILD / "assets/editorial/editorial.css").read_text()
    assert ".home-hero" in css and ".latest-post" in css
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
    assert list(zip(page.latest, page.latest_titles)) == expected[:LATEST_COUNT], "Latest list must match the refreshed feed"
    for path in ["building/index.html", "writing/index.html", "lattice-graph.html"]:
        assert (BUILD / path).is_file(), f"Preserve existing URL: {path}"
    assert (BUILD / "CNAME").read_text().strip() == "weijian.ai"
    print("Static homepage verified: current feed, featured essays, follow links, images, fonts, archive URLs and domain.")


if __name__ == "__main__":
    main()
