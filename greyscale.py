import glob
from PIL import Image

i = 0
for c in glob.glob("/home3/bkorkor/distracteddriver/data/train/*"):
    print c
    for imagefile in glob.glob(c + "/*"):
        if i % 1000 == 0:
            print i
        i += 1
        img = Image.open(imagefile)
        img.convert('L').save(imagefile)
