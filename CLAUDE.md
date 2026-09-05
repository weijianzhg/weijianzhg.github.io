# Project notes

Personal site at https://weijian.ai. The homepage is standalone HTML and CSS; Jekyll builds the site and writing archive.

- Edit the homepage in `_pages/about.html` and its styles in `assets/editorial/`.
- Keep the existing `/writing/`, `/building/` and `/lattice-graph.html` URLs working.
- Run `bundle exec jekyll build --lsi`, then `python3 bin/check-editorial-build.py` to check the generated site. Run `npx prettier . --check` for formatting.
- `main` is the source branch. GitHub Actions publishes to `gh-pages` and refreshes feeds hourly.
- `./bin/update-feed-cache` refreshes local RSS snapshots. `./bin/publish-feeds` also commits and pushes them; use it when publishing feed updates is requested.

See README.md for setup and editing instructions.
