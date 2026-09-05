# Weijian Zhang

Source for [weijian.ai](https://weijian.ai), my personal website about AI architecture and building reliable systems.

The homepage is static HTML and CSS with local fonts and no JavaScript. Jekyll builds the site, including the writing archive and existing project pages.

## Editing

- `_pages/about.html`: homepage copy, selected projects and featured essays.
- `assets/editorial/`: homepage styles, icons and fonts.
- `_pages/blog.md` and `_pages/building.md`: writing archive and work page.
- `lattice-graph.html`: interactive mental model graph.
- `_config.yml`: site settings and external writing sources.

The archive uses the layouts, includes and styles inherited from [al-folio](https://github.com/alshedivat/al-folio).

## Run locally

Use Ruby with Bundler (CI uses Ruby 3.2.2), ImageMagick and Python 3. Node.js is only needed for formatting.

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
