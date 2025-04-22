from PIL import Image, ImageDraw

# Load the image
image_path = "/home/llmnav/lianqi/SoM/SYNTH-exp/composed_image_0000.jpg"
image = Image.open(image_path).convert("RGB")

import pdb
pdb.set_trace()
# Define bounding box coordinates
bbox = (40, 20, 140, 210)  # (x1, y1, x2, y2)

# Draw the bounding box
draw = ImageDraw.Draw(image)
draw.rectangle(bbox, outline="red", width=3)

# Show the image with the bounding box
image.save("/home/llmnav/lianqi/SoM/SYNTH-exp/composed_image_0000_bbox.jpg")