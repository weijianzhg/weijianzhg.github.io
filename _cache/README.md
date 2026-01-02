# External Feed Cache

This directory contains cached RSS/Atom feeds from external sources to ensure reliable builds even when external services are unavailable.

## How It Works

1. The cached feeds are manually updated and committed to the repository
2. During site builds, the `external-posts.rb` plugin uses the cached feed instead of fetching from the URL
3. This prevents build failures when external services (like Substack) block or rate-limit requests

## Manual Update

To manually update the cache locally, run:

```bash
./bin/update-feed-cache
```

Or manually with curl:

```bash
mkdir -p _cache
curl -H "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36" \
     --connect-timeout 10 \
     --max-time 30 \
     -f \
     -o _cache/substack.com_feed.xml \
     https://notesfromzero.substack.com/feed
```

After updating, commit the changes:

```bash
git add _cache/substack.com_feed.xml
git commit -m "chore: update external feed cache"
git push
```

## Files

- `substack.com_feed.xml` - Cached RSS feed from Notes From Zero Substack
