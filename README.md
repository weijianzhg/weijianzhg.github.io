# Weijian Zhang

Source for [weijian.ai](https://weijian.ai), my personal website: writing about AI, and building it for finance.

Every page is static HTML and CSS with local fonts and no JavaScript. The homepage leads with writing and places to follow, and features one project, Hedge Layer. Jekyll builds the site, including the writing archive and work page.

## Editing

- `_pages/about.html`: homepage copy and section order.
- `_data/start_here.yml`: the highlighted essays on the homepage and writing page. Update `FEATURED` in `bin/check-editorial-build.py` to match.
- `_data/channels.yml`: the Substack, X and LinkedIn cards under “Where I write”.
- `_data/projects.yml`: the main project (Hedge Layer) shown on the homepage and at the top of the work page.
- `assets/editorial/work/`: project screenshots as 16:10 WebP, at 800px and 1600px wide (`<name>-800.webp`, `<name>-1600.webp`).
- `_pages/building.html` and `_pages/writing.html`: work page (main project, then other projects) and writing archive.
- `_layouts/editorial.liquid` and `_includes/editorial-*.liquid`: shared page shell, header, footer and post rows.
- `assets/editorial/`: styles, icons and fonts for every page. `og.png` (1200×630) is the preview image for shared links and repeats the homepage headline.
- `lattice-graph.html`: interactive mental model graph.
- `_config.yml`: site settings and external writing sources.

## Run locally

Use Ruby with Bundler (CI uses Ruby 3.3.6 and the committed `Gemfile.lock`) and Python 3. Node.js is only needed for formatting.

```sh
bundle install
bundle exec jekyll serve --livereload
```

Open [localhost:4000](http://localhost:4000). The committed feed cache lets you build without fetching new posts.

Before pushing:

```sh
JEKYLL_ENV=production bundle exec jekyll build
python3 bin/check-editorial-build.py
npm ci
npx prettier . --check
```

## Publishing and writing updates

Push site changes to `main` (including merged pull requests), or manually run **Deploy site** on `main` in [GitHub Actions](.github/workflows/deploy.yml), to publish. The workflow builds and checks the site, then updates `gh-pages`. GitHub Pages serves it at **weijian.ai**, configured by `CNAME`. Pull requests do not trigger the deployment workflow.

Each deployment run refreshes both Substack feeds; there are no scheduled deployments or feed refreshes. The homepage shows the three highlighted essays, then the five most recent other Notes from Zero posts, without duplicates. If a feed is unavailable or invalid, the build keeps the last successful cache.

To refresh feeds locally, run `./bin/update-feed-cache` from the repository root. To refresh, commit and push the cache in one step, run `./bin/publish-feeds`.

## License

Content © Weijian Zhang. The original al-folio theme is covered by [LICENSE](LICENSE). Font licenses are included in `assets/editorial/fonts/`.
