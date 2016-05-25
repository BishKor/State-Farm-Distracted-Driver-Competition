import glob
from PIL import Image

i = 0
for imagefile in glob.glob("/home3/bkorkor/distracteddriver/data/test/*"):
    if i % 1000 == 0:
        print i
    i += 1
    img = Image.open(imagefile)
    img.convert('L').save(imagefile)
