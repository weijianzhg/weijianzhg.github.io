# Weijian Zhang

Source for [weijian.ai](https://weijian.ai), my personal website about building reliable AI systems for finance.

Every page is static HTML and CSS with local fonts and no JavaScript. Jekyll builds the site, including the writing archive and work page.

## Editing

- `_pages/about.html`: homepage copy, selected projects and featured essays.
- `_pages/building.html` and `_pages/writing.html`: work page and writing archive.
- `_layouts/editorial.liquid` and `_includes/editorial-*.liquid`: shared page shell, header, footer and post rows.
- `assets/editorial/`: styles, icons and fonts for every page.
- `lattice-graph.html`: interactive mental model graph.
- `_config.yml`: site settings and external writing sources.

## Run locally

Use Ruby with Bundler (CI uses Ruby 3.3.6 and the committed `Gemfile.lock`), ImageMagick and Python 3. Node.js is only needed for formatting.

```sh
bundle install
bundle exec jekyll serve --livereload
```

Open [localhost:4000](http://localhost:4000). The committed feed cache lets you build without fetching new posts.

Before pushing:

```sh
JEKYLL_ENV=production bundle exec jekyll build --lsi
python3 bin/check-editorial-build.py
npm ci
npx prettier . --check
```

## Publishing and writing updates

Push to `main` to publish. [GitHub Actions](.github/workflows/deploy.yml) builds and checks the site, then updates `gh-pages`. GitHub Pages serves it at **weijian.ai**, configured by `CNAME`. Pull requests build without publishing.

The workflow also runs hourly to refresh both Substack feeds. The homepage shows three recent Notes from Zero posts alongside two curated essays, without duplicates. If a feed is unavailable or invalid, the build keeps the last successful cache. GitHub may delay scheduled runs.

To refresh feeds locally, run `./bin/update-feed-cache` from the repository root. To refresh, commit and push the cache in one step, run `./bin/publish-feeds`.

## License

Content © Weijian Zhang. The original al-folio theme is covered by [LICENSE](LICENSE). Font licenses are included in `assets/editorial/fonts/`.
