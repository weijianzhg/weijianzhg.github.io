# Project Notes

Jekyll site (al-folio theme) deployed to GitHub Pages.

## Common Tasks

- **Update and publish feeds**: Run `./bin/publish-feeds`. This fetches RSS feeds from Substack, commits any cache changes, and pushes to trigger a deploy. No research needed — just run the script.
- **Update feeds only (no push)**: Run `./bin/update-feed-cache`.

## Feed System

- External RSS sources are defined in `_config.yml` under `external_sources`.
- Cached feed XML lives in `_cache/`.
- `_plugins/external-posts.rb` reads the cache during Jekyll build to create blog posts.
- The deploy workflow (`.github/workflows/deploy.yml`) auto-rebuilds when `_cache/**` changes.
