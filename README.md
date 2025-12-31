# Weijian Zhang's Personal Website

Personal website and blog built with Jekyll, based on the [al-folio](https://github.com/alshedivat/al-folio) theme.

🌐 **Live site:** [weijianzhg.com](https://weijianzhg.com)

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
