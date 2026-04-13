import numpy as np
from PIL import Image
from pathlib import Path

input_dir  = Path("generated_images")
output_dir = Path("generated_images_filtered")
output_dir.mkdir(exist_ok=True)

BRIGHTNESS_THRESHOLD = 20    # mean pixel value out of 255
                              # pure black = 0, good X-ray = typically 30-80+

kept    = 0
removed = 0

for img_path in sorted(input_dir.glob("*.png")):
    img        = Image.open(img_path).convert("L")
    mean_pixel = np.array(img).mean()

    if mean_pixel >= BRIGHTNESS_THRESHOLD:
        img.save(output_dir / img_path.name)
        kept += 1
    else:
        removed += 1

print(f"Kept   : {kept}")
print(f"Removed: {removed}")