#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const https = require('https');

// Check if sharp is installed, install if not
let sharp;
try {
  sharp = require('sharp');
} catch (e) {
  console.log('Installing sharp...');
  require('child_process').execSync('npm install sharp', { stdio: 'inherit' });
  sharp = require('sharp');
}

const MOMENTS_DIR = path.join(__dirname, '../assets/images/moments');
const THUMB_DIR = path.join(__dirname, '../assets/images/moments/thumbs');
const THUMB_2X_DIR = path.join(__dirname, '../assets/images/moments/thumbs-2x');

// Create directories if they don't exist
[THUMB_DIR, THUMB_2X_DIR].forEach(dir => {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
});

// Get all JPG files
const files = fs.readdirSync(MOMENTS_DIR).filter(f => f.endsWith('.jpg'));

console.log(`Found ${files.length} images to optimize...`);

let processed = 0;

files.forEach(file => {
  const inputPath = path.join(MOMENTS_DIR, file);
  const thumbPath = path.join(THUMB_DIR, file);
  const thumb2xPath = path.join(THUMB_2X_DIR, file);

  // 1x thumbnail (220px)
  sharp(inputPath)
    .resize(220, 220, {
      fit: 'cover',
      position: 'center'
    })
    .jpeg({ quality: 75, progressive: true })
    .toFile(thumbPath, (err, info) => {
      if (err) {
        console.error(`Error processing ${file}:`, err);
      } else {
        console.log(`✓ ${file} → thumbs/${file} (${info.size} bytes)`);
        processed++;
        if (processed === files.length * 2) {
          console.log('\n✓ All images optimized!');
          console.log(`\nNext: Update moments.html to use srcset with thumbs/ and thumbs-2x/`);
        }
      }
    });

  // 2x thumbnail (440px for retina)
  sharp(inputPath)
    .resize(440, 440, {
      fit: 'cover',
      position: 'center'
    })
    .jpeg({ quality: 75, progressive: true })
    .toFile(thumb2xPath, (err, info) => {
      if (err) {
        console.error(`Error processing ${file} (2x):`, err);
      } else {
        console.log(`✓ ${file} → thumbs-2x/${file} (${info.size} bytes)`);
        processed++;
        if (processed === files.length * 2) {
          console.log('\n✓ All images optimized!');
          console.log(`\nNext: Update moments.html to use srcset with thumbs/ and thumbs-2x/`);
        }
      }
    });
});
