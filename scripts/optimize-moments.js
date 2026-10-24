#!/usr/bin/env node
/**
 * Regenerates every Moments image derivative from the originals:
 *
 *   thumbs/     220x220 cover   — gallery grid at 1x
 *   thumbs-2x/  440x440 cover   — gallery grid at 2x (grid cell is 214 CSS px, capped)
 *   large/      2560 long edge  — lightbox (90vw/80vh, no upscale, q80)
 *
 * Originals in moments/ stay untouched as masters.
 *
 *   node scripts/optimize-moments.js
 *   node scripts/optimize-moments.js --only jit-ray-chowdhury-with-tesla-optimus.jpg
 */
const fs = require('fs');
const path = require('path');
const sharp = require('sharp');

const ROOT = path.join(__dirname, '..');
const SRC_DIR = path.join(ROOT, 'assets/images/moments');
const VARIANTS = [
    { dir: 'thumbs', width: 220, height: 220, quality: 75 },
    { dir: 'thumbs-2x', width: 440, height: 440, quality: 75 },
    { dir: 'large', width: 2560, height: null, quality: 80 },
];

const onlyIdx = process.argv.indexOf('--only');
const only = onlyIdx > -1 ? process.argv[onlyIdx + 1] : null;

async function main() {
    const files = fs.readdirSync(SRC_DIR)
        .filter((f) => /\.(jpe?g|png)$/i.test(f))
        .filter((f) => !only || f === only);

    if (!files.length) {
        console.error(only ? `No such file: ${only}` : 'No source images found.');
        process.exit(1);
    }

    for (const v of VARIANTS) fs.mkdirSync(path.join(SRC_DIR, v.dir), { recursive: true });
    console.log(`Processing ${files.length} image(s)...\n`);

    let before = 0;
    let after = 0;

    for (const file of files) {
        const input = path.join(SRC_DIR, file);
        const meta = await sharp(input).metadata();
        const original = fs.statSync(input).size;
        before += original;

        const parts = [];
        for (const v of VARIANTS) {
            const out = path.join(SRC_DIR, v.dir, file);
            const pipeline = sharp(input).rotate().resize(v.width, v.height, {
                fit: v.height ? 'cover' : 'inside',
                position: 'centre',
                withoutEnlargement: true,
            });
            if (/\.jpe?g$/i.test(file)) pipeline.jpeg({ quality: v.quality, progressive: true, mozjpeg: true });
            else pipeline.png({ compressionLevel: 9 });
            await pipeline.toFile(out);

            const bytes = fs.statSync(out).size;
            after += bytes;
            const dim = await sharp(out).metadata();
            parts.push(`${v.dir} ${dim.width}x${dim.height} ${(bytes / 1024).toFixed(0)}K`);
        }

        console.log(`${file}\n  ${meta.width}x${meta.height} ${(original / 1024).toFixed(0)}K  ->  ${parts.join('  |  ')}`);
    }

    console.log(`\nOriginals ${(before / 1048576).toFixed(1)} MB, derivatives ${(after / 1048576).toFixed(1)} MB.`);
}

main().catch((err) => {
    console.error(err);
    process.exit(1);
});
