# jitrc.github.io

Personal site (Jekyll) published at <https://www.jitrc.com> by GitHub Pages. This file is
the portable source of truth for agents working in this repo.

## Layout

| Path | What it is |
|---|---|
| `index.html`, `moments.html`, `projects.html`, `tags.html` | Top-level pages |
| `_layouts/` | `default` → `page` → `post` / `nutshell` |
| `_includes/` | Header, nav, footer, `style.scss` (Sass entrypoint) |
| `_sass/` | Indented-syntax partials, imported by `_includes/style.scss` |
| `_nutshell/` | Cheat-sheet notes, rendered at `/nutshell/<slug>/` |
| `_data/moments.yml` | Moments gallery content |
| `scripts/optimize-moments.js` | Regenerates thumbnails with `sharp` |
| `_config.yml` | Site config; nav flags (`blog`, `moments`, `nutshell`, …) |

Nav links are gated by booleans in `_config.yml` (e.g. `show_nutshell: true`) and rendered
in `_includes/nav.html`. A flag must not share a name with a collection: `nutshell: true`
would be shadowed by the `_nutshell` collection, so `site.nutshell` becomes the document
array and the nav check silently fails.

Anything not matched by `exclude:` in `_config.yml` is copied to the web root, so new
repo-level files (notes, scripts, agent instructions) need an entry there or they become
public.

## Local preview

System Ruby (2.6 on macOS) cannot install the pinned `github-pages` gem. Use Homebrew Ruby
with plain Jekyll 4:

```bash
brew install ruby
export PATH=/opt/homebrew/opt/ruby/bin:/opt/homebrew/lib/ruby/gems/4.0.0/bin:$PATH
gem install jekyll jekyll-feed jekyll-seo-tag jekyll-gist jemoji jekyll-mentions --no-document
JEKYLL_NO_BUNDLER_REQUIRE=true jekyll serve --livereload   # http://127.0.0.1:4000
```

- `JEKYLL_NO_BUNDLER_REQUIRE=true` is required, or Jekyll tries to resolve the repo's
  `github-pages` Gemfile and dies with `Could not find gem 'github-pages'`.
- Plugins must be listed under `plugins:` in `_config.yml`; Jekyll 4 ignores the legacy
  `gems:` key and `{% seo %}` then fails as an unknown tag.
- Dart Sass (Jekyll 4) rejects tab indentation. `.sass` partials must use spaces, even
  though Ruby Sass on GitHub Pages tolerates tabs.
- `_site/` and `.jekyll-cache/` are build artifacts.

Exact GitHub Pages build (pixel-identical) needs Ruby 3.x plus
`bundle install && bundle exec jekyll serve`; `github-pages` does not support Ruby 4.

## Nutshell notes

`_nutshell/*.md` render at `/nutshell/<slug>/` and are listed newest-first on `/nutshell/`.
These are reference cheat sheets, not essays — someone should skim a table and copy a
command.

### Non-negotiable: research first

Never write a note from memory, and never trust the existing note — it may be the stale
thing you were asked to fix. Before drafting:

1. **Go to primary sources.** Official spec/docs, changelog, release notes, migration
   guide. Secondary blog posts are leads only; verify at the source.
2. **Establish current state.** Latest stable version, current release candidate, what
   shipped recently, and the date checked. If docs lag a release, say so and prefer the
   changelog / release notes / actual API.
3. **Find what bites people.** Search issues and discussions for breaking changes,
   deprecations, known bugs, and the workarounds that landed.
4. **Verify snippets.** Run the command or import where feasible. Check package names and
   entrypoints against the current release — renames are common (`FastMCP` → `MCPServer`
   is one). Prefer measured numbers from a real run over borrowed ones.

State the revision/date verified against inside the note. For fast-moving topics
(protocols, model APIs, agent harnesses), a note describing last year's API without saying
so is worse than no note.

### Shape

1. `## What it does` — one short paragraph.
2. `## Why it exists` — 3 bullets on the problem solved.
3. A diagram when the mental model is not obvious.
4. `## Cheat sheet: ...` — table of commands/verbs/methods and what they do.
5. Sections for the parts people get wrong (cost, transports, state, security).
6. `## Snippets` — smallest working server/client/config/CLI commands.
7. `## Typical use cases` — scenario → what you'd expose/configure.
8. `## Gotchas` — short, non-obvious, each actionable.
9. Optional boundary section ("where it sits next to X").

Prefer tables over prose. If a paragraph would be a table row, make it a table row.

### Front matter and styling

```yaml
---
layout: nutshell
title: "Topic in a Nutshell"
date: 2026-10-23
summary: "One line shown on the index and under the title."
tags: [topic, area]
---
```

- `<figure class="diagram"><svg …>…</svg><figcaption>…</figcaption></figure>` — responsive
  inline SVG, no JS. Add `role="img"` and a `<title>`.
- `<div class="callout"><strong>Title</strong> …</div>` — aside for version traps.
- Tables are styled automatically; code in the first column stays on one line.
- Markdown and raw HTML both render (kramdown).

### Verify and publish

```bash
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:4000/nutshell/<slug>/
```

Check the new sections render before committing. Commit inside this repo as
`jit.ray.c@gmail.com`; write a message that states what changed and why; do not push unless
asked.

## Social cards

Every essay and note gets a 1200×630 card at `assets/images/og/<slug>.png`, used as
`og:image` and `twitter:image` (so shares unfurl as a large card instead of a text link).

```bash
node scripts/og-cards.js            # render all cards, add image: front matter if missing
node scripts/og-cards.js --dry-run  # show what would happen
```

Run it after changing a `title`, `description` or `summary`, and after adding a file. It
writes `image: { path, width, height, alt }` into the front matter once; PNGs are always
rewritten. Needs Chrome or Chromium locally (`CHROME_BIN` to override the path). The script
is excluded from the published site; the PNGs are not.

