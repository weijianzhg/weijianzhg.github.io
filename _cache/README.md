# Feed cache

These RSS snapshots let the site build when Substack is unavailable:

- `notesfromzero.substack.com_feed.xml`
- `latticeworkofmodels.substack.com_feed.xml`

The deploy workflow refreshes them on each build and on an hourly schedule. GitHub Actions retains successful snapshots between runs; the committed files are the fallback. Invalid responses never replace a valid cache.

`_plugins/external-posts.rb` reads these files during the Jekyll build. To refresh them locally, run `./bin/update-feed-cache` from the repository root. See the [README](../README.md) for publishing instructions.
