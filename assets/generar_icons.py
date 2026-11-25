from PIL import Image

img = Image.open("assets/icon.png").convert("RGBA")
img.save(
    "assets/icon.ico",
    format="ICO",
    sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
)