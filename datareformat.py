import random
import numpy as np
from PIL import Image
from skimage import color

image_list = open('data/driver_imgs_list.csv')

images = image_list.readlines()[1:]
random.shuffle(images)

trainingset = np.empty((len(images), 480, 640))
traininglabels = np.empty((len(images), 10))

for index, image in enumerate(images):
    if index % 1000 == 0:
        print(float(index)/len(images))
    # Convert scale from integers [0, 255] to floats [0,1]
    colorimage = np.asarray(Image.open("data/imgs/train/"+image[5:7]+"/"+image[8:].rstrip("\n")))/np.float32(256)
    trainingset[index] = color.rgb2grey(colorimage)
    traininglabels[index][int(image[6:7])] = 1.

np.save("training_labels", traininglabels)
np.save("training_images", trainingset)
