#!/usr/bin/env node
/**
 * Renders 1200x630 social cards for every essay in _posts/ and every note in
 * _nutshell/, writes them to assets/images/og/, and adds an `image:` key to the
 * front matter so jekyll-seo-tag emits og:image + twitter:card=summary_large_image.
 *
 * Re-run after changing a title, description or summary. Safe to re-run: PNGs are
 * rewritten, front matter is only touched when `image:` is missing.
 *
 *   node scripts/og-cards.js            # all items
 *   node scripts/og-cards.js --dry-run  # report only
 */
const fs = require('fs');
const os = require('os');
const path = require('path');
const { execFileSync } = require('child_process');

const ROOT = path.resolve(__dirname, '..');
const OUT_DIR = path.join(ROOT, 'assets/images/og');
const DRY = process.argv.includes('--dry-run');

function findChrome() {
    const candidates = [
        process.env.CHROME_BIN,
        '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
        '/Applications/Chromium.app/Contents/MacOS/Chromium',
        '/usr/bin/google-chrome',
        '/usr/bin/chromium',
        '/usr/bin/chromium-browser',
    ].filter(Boolean);
    const found = candidates.find((p) => fs.existsSync(p));
    if (!found) throw new Error('No Chrome/Chromium found. Set CHROME_BIN.');
    return found;
}

function readFrontMatter(file) {
    const raw = fs.readFileSync(file, 'utf8');
    const match = raw.match(/^---\n([\s\S]*?)\n---\n?/);
    if (!match) return null;
    const fm = match[1];
    const get = (key) => {
        const m = fm.match(new RegExp(`^${key}:\\s*(.+)$`, 'm'));
        return m ? m[1].trim().replace(/^["']|["']$/g, '') : null;
    };
    const tagsRaw = get('tags');
    const tags = tagsRaw
        ? tagsRaw.replace(/^\[|\]$/g, '').split(',').map((t) => t.trim()).filter(Boolean)
        : [];
    return { fm, body: raw.slice(match[0].length), title: get('title'), date: get('date'), tags, description: get('description') || get('summary'), image: get('image') };
}

function trim(text, limit) {
    if (!text || text.length <= limit) return text || '';
    const sentence = text.match(/^.*?[.!?](?=\s|$)/);
    if (sentence && sentence[0].length <= limit) return sentence[0];
    const cut = text.slice(0, limit);
    const clause = Math.max(cut.lastIndexOf(', '), cut.lastIndexOf('; '), cut.lastIndexOf(': '));
    if (clause > limit * 0.55) return cut.slice(0, clause).replace(/[,;:.\s]+$/, '') + '\u2026';
    return cut.slice(0, cut.lastIndexOf(' ')).replace(/[,;:.\s]+$/, '') + '\u2026';
}

const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');

function titleSize(title) {
    if (title.length <= 32) return 76;
    if (title.length <= 52) return 64;
    if (title.length <= 80) return 52;
    if (title.length <= 110) return 44;
    return 38;
}

function html({ title, description, meta, kind }) {
    return `<!DOCTYPE html><html><head><meta charset="utf-8"><style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    width: 1200px; height: 630px; background: #fff; color: #222;
    font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
    padding: 64px 72px 56px; display: flex; flex-direction: column;
    justify-content: space-between; position: relative; overflow: hidden;
  }
  .rule { position: absolute; top: 0; left: 0; width: 100%; height: 8px; background: #4b0082; }
  .top { display: flex; justify-content: space-between; align-items: baseline; }
  .brand { font-size: 21px; letter-spacing: .18em; text-transform: uppercase; color: #222; font-weight: 600; }
  .kind { font-size: 19px; letter-spacing: .16em; text-transform: uppercase; color: #aaa; }
  h1 { font-size: ${titleSize(title)}px; line-height: 1.1; letter-spacing: -.018em; font-weight: 700; max-height: 4.4em; overflow: hidden; }
  .desc { font-size: 24px; line-height: 1.4; color: #777; max-height: 2.8em; overflow: hidden; max-width: 980px; }
  .foot { border-top: 1px solid #ededed; padding-top: 18px; display: flex; justify-content: space-between; font-size: 19px; color: #aaa; letter-spacing: .02em; }
  .tags { color: #4b0082; }
</style></head><body>
  <div class="rule"></div>
  <div class="top"><span class="brand">jitrc.com</span><span class="kind">${esc(kind)}</span></div>
  <div>
    <h1>${esc(title)}</h1>
    ${description ? `<p class="desc" style="margin-top:26px">${esc(description)}</p>` : ''}
  </div>
  <div class="foot"><span>${esc(meta)}</span><span class="tags">${esc(tagsLine(kind))}</span></div>
</body></html>`;
}

function tagsLine(kind) {
    return kind === 'Nutshell' ? 'cheat sheet' : 'essay';
}

function metaLine(front) {
    const d = front.date ? new Date(front.date) : new Date();
    const label = isNaN(d) ? '' : d.toLocaleDateString('en-US', { month: 'short', year: 'numeric' });
    const tags = front.tags.slice(0, 3).join(' · ');
    return [label, tags].filter(Boolean).join('  ·  ');
}

function slugOf(file) {
    return path.basename(file, path.extname(file)).replace(/^\d{4}-\d{2}-\d{2}-/, '');
}

function collect() {
    const items = [];
    const add = (dir, kind, collection) => {
        if (!fs.existsSync(dir)) return;
        for (const name of fs.readdirSync(dir).filter((f) => /\.(md|markdown)$/.test(f))) {
            const file = path.join(dir, name);
            const front = readFrontMatter(file);
            if (!front || !front.title) continue;
            items.push({ file, kind, collection, front, slug: slugOf(name) });
        }
    };
    add(path.join(ROOT, '_posts'), 'Writing', 'posts');
    add(path.join(ROOT, '_nutshell'), 'Nutshell', 'nutshell');
    return items;
}

function main() {
    const chrome = DRY ? null : findChrome();
    if (!DRY) fs.mkdirSync(OUT_DIR, { recursive: true });
    const items = collect();
    if (!items.length) return console.log('No items found.');

    for (const item of items) {
        const rel = `/assets/images/og/${item.slug}.png`;
        const out = path.join(ROOT, rel.replace(/^\//, ''));
        console.log(`${DRY ? '[dry] ' : ''}${item.collection}/${item.slug} -> ${rel}`);

        if (!DRY) {
            const tmp = path.join(os.tmpdir(), `og-${item.slug}.html`);
            fs.writeFileSync(tmp, html({
                title: item.front.title,
                description: trim(item.front.description, 150),
                meta: metaLine(item.front),
                kind: item.kind,
            }));
            execFileSync(chrome, [
                '--headless', '--disable-gpu', '--hide-scrollbars', '--no-sandbox',
                '--force-device-scale-factor=1', '--window-size=1200,630',
                '--default-background-color=ffffffff', '--virtual-time-budget=1500',
                `--screenshot=${out}`, `file://${tmp}`,
            ], { stdio: 'ignore' });
            fs.unlinkSync(tmp);
        }

        if (!item.front.image) {
            const alt = `${item.front.title} \u2014 jitrc.com`.replace(/"/g, '\\"');
            const block = [
                'image:',
                `  path: ${rel}`,
                '  width: 1200',
                '  height: 630',
                `  alt: "${alt}"`,
            ];
            const fmLines = item.front.fm.split('\n');
            const dateIdx = fmLines.findIndex((l) => /^date:/.test(l));
            const insertAt = dateIdx >= 0 ? dateIdx + 1 : fmLines.length;
            fmLines.splice(insertAt, 0, ...block);
            const raw = fs.readFileSync(item.file, 'utf8');
            const body = raw.slice(raw.indexOf('\n---', 3) + 4);
            if (!DRY) fs.writeFileSync(item.file, `---\n${fmLines.join('\n')}\n---\n${body}`);
            console.log(`  + front matter: image (${rel})`);
        }
    }
}

main();
