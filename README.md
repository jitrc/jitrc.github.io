# jitrc.github.io

Personal site (Jekyll) published at <https://www.jitrc.com> by GitHub Pages.

## Local preview

GitHub Pages pins old gem versions, and macOS ships Ruby 2.6, so the repo's `Gemfile`
(`github-pages`) will not install on a stock Mac. The fastest reliable route is a modern
Ruby plus plain Jekyll 4:

```bash
brew install ruby
export PATH=/opt/homebrew/opt/ruby/bin:/opt/homebrew/lib/ruby/gems/4.0.0/bin:$PATH

gem install jekyll jekyll-feed jekyll-seo-tag jekyll-gist jemoji jekyll-mentions --no-document

# Jekyll auto-loads the Gemfile; this tells it to use the gems installed above instead.
JEKYLL_NO_BUNDLER_REQUIRE=true jekyll serve --livereload
```

Open <http://127.0.0.1:4000>. `--livereload` refreshes the browser on save.

Notes:
- If you *do* want the exact GitHub Pages build, use Ruby 3.x and
  `bundle install && bundle exec jekyll serve` (the `github-pages` gem does not support Ruby 4).
- The Sass partials are indented `.sass` using spaces — Dart Sass (Jekyll 4) rejects tabs,
  even though Ruby Sass (GitHub Pages) accepts them.
- `_site/` and `.jekyll-cache/` are build artifacts.

## Nutshell notes

`_nutshell/*.md` render at `/nutshell/<filename>/` and are listed on `/nutshell/`. New note:

```yaml
---
layout: nutshell
title: "Topic in a Nutshell"
date: 2026-10-23
summary: "One line shown on the index."
tags: [topic]
---
```

Keep them cheat-sheet shaped: what it does, why it exists, the verbs/commands that matter,
copy-paste snippets, typical use cases, gotchas. Inline SVG in a `.diagram` figure and a
`.callout` div are styled for this section.

**Research before writing or updating.** Notes age fast: check the current spec, changelog,
and release notes for the topic (primary sources only), and state the revision/date you
verified against. Call out deprecations, known issues, fixes, and migration steps between
versions — a note that describes last year's API without saying so is worse than none.
