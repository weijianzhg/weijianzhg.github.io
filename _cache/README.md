# External Feed Cache

This directory contains cached RSS/Atom feeds from external sources to ensure reliable builds even when external services are unavailable.

## How It Works

1. The `update-feed-cache.yml` GitHub Action runs every 6 hours to fetch fresh feeds
2. The cached feeds are committed to the repository
3. During site builds, the `external-posts.rb` plugin uses the cached feed instead of fetching from the URL
4. This prevents build failures when external services (like Substack) block or rate-limit GitHub Actions

## Manual Update

To manually update the cache locally:

```bash
curl -H "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36" \
     -o _cache/substack.com_feed.xml \
     https://notesfromzero.substack.com/feed
```

Or trigger the GitHub Action manually from the Actions tab.

## Files

- `substack.com_feed.xml` - Cached RSS feed from Notes From Zero Substack
