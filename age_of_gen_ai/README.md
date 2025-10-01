# Age of Gen AI Presentation

This presentation is built using [Marp](https://marp.app/), a Markdown-based presentation ecosystem.

## Viewing the Presentation

### Option 1: Using Marp CLI (Recommended)

1. Install Marp CLI:
```bash
npm install -g @marp-team/marp-cli
```

2. Generate HTML:
```bash
marp presentation.md -o index.html
```

3. Open `index.html` in your browser

### Option 2: Live Preview with VS Code

1. Install the [Marp for VS Code](https://marketplace.visualstudio.com/items?itemName=marp-team.marp-vscode) extension
2. Open `presentation.md` in VS Code
3. Click the "Open Preview" button (or press `Ctrl+Shift+V` / `Cmd+Shift+V`)

### Option 3: Export to PDF

```bash
marp presentation.md --pdf
```
### Option 4: Export as PowerPoint

```bash
marp presentation.md --pptx -o presentation.pptx
```

## Themes

**Recommended themes:** `default` or `uncover`

To switch themes, edit line 3 in `presentation.md`:
```markdown
theme: default  # or: uncover, gaia
```

Then regenerate the outputs.

## Editing

Simply edit `presentation.md` - it's just Markdown!

## Adding Images

Use standard Markdown image syntax with Marp directives:

```markdown
# Basic image
![alt text](path/to/image.jpg)

# Sized image
![w:600](image.jpg)          # width
![h:400](image.jpg)          # height
![w:800 h:600](image.jpg)    # both

# Background image (full slide)
![bg](image.jpg)
![bg opacity:0.5](image.jpg)

# Image on side with text
![bg right:50%](diagram.png)

# Two images side by side
![w:400](image1.jpg) ![w:400](image2.jpg)
```

**Tip:** Store images in an `images/` folder and reference them as `images/filename.jpg`
