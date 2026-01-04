# Extension Icons

Place your extension icons here:

- `icon16.png` - 16x16 pixels
- `icon32.png` - 32x32 pixels
- `icon48.png` - 48x48 pixels
- `icon128.png` - 128x128 pixels

## Creating Icons

You can use the following Python script to generate simple placeholder icons:

```python
from PIL import Image, ImageDraw

sizes = [16, 32, 48, 128]
colors = {
    'bg': (139, 92, 246),  # Purple
    'fg': (255, 255, 255)  # White
}

for size in sizes:
    img = Image.new('RGBA', (size, size), colors['bg'])
    draw = ImageDraw.Draw(img)

    # Draw chess knight silhouette (simplified)
    margin = size // 8
    draw.ellipse([margin, margin, size-margin, size-margin], fill=colors['fg'])

    img.save(f'icon{size}.png')
```

Or use any image editor to create your custom chess-themed icons.
