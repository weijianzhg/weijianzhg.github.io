## Personal homepage and publishing

The homepage uses the approved editorial design: standalone HTML and CSS, with locally hosted fonts. It has no client framework or JavaScript requirement. The existing Jekyll build still generates the writing archive and preserves its URLs.

Pushes to `main` publish through `.github/workflows/deploy.yml` to the `gh-pages` branch. GitHub Pages serves that branch at **https://weijian.ai**. `CNAME` preserves the domain. Pull requests and manual builds on other branches validate without publishing.

The same workflow runs at minute 17 of each hour to refresh the public blog feeds and rebuild the static site. GitHub may delay scheduled runs. If a feed request fails or returns invalid XML, the last successful GitHub Actions feed cache is retained, with the committed cache as a fallback. Two featured essays remain curated; the three latest posts come from Notes from Zero and exclude those highlights. No server or runtime RSS proxy is used on GitHub Pages.

Homepage source: `_pages/about.html`. Styles and fonts: `assets/editorial/`. Build verification: `python3 bin/check-editorial-build.py` after `bundle exec jekyll build`.

---

# Weijian Zhang's Personal Website

Personal website and blog built with Jekyll, based on the [al-folio](https://github.com/alshedivat/al-folio) theme.

🌐 **Live site:** [weijian.ai](https://weijian.ai)

## Local Development

### Option 1: Docker (Recommended)

The easiest way to run the site locally:

```bash
docker compose up
```

Then open [http://localhost:8080](http://localhost:8080) in your browser.

The site will auto-reload when you make changes.

### Option 2: Native Ruby/Jekyll

If you prefer running without Docker:

1. **Install dependencies:**

   ```bash
   # macOS
   brew install ruby imagemagick

   # Add Ruby to PATH (add to ~/.zshrc)
   export PATH="/opt/homebrew/opt/ruby/bin:$PATH"
   ```

2. **Install gems:**

   ```bash
   bundle install
   ```

3. **Run the site:**

   ```bash
   bundle exec jekyll serve --livereload
   ```

   Then open [http://localhost:4000](http://localhost:4000)

## Project Structure

```
├── _pages/           # Site pages (about, blog, 404)
├── _posts/           # Blog posts
├── _config.yml       # Site configuration
├── assets/
│   ├── img/          # Images
│   ├── css/          # Stylesheets
│   └── js/           # JavaScript
└── _sass/            # SCSS source files
```

## Writing Blog Posts

Create a new file in `_posts/` with the format `YYYY-MM-DD-title.md`:

```markdown
---
layout: post
title: Your Post Title
date: 2024-01-15
description: A short description
tags: tag1, tag2
categories: category-name
---

Your content here...
```

## License

Content © Weijian Zhang. Theme based on [al-folio](https://github.com/alshedivat/al-folio) (MIT License).
