print("NEW FILE RUNNING")
import os
# PIL = Python Imaging Library
import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import Image

train_path = "dataset/train"
classes = os.listdir(train_path)

print(len(classes))
first_class = classes[0]

print(first_class)


files = os.listdir("dataset/train/freshapples")

print("Total Images:", len(files))
print("First 5 Images:", files[:5])
first_image = files[0]
image_path = "dataset/train/freshapples/" + first_image

print(image_path)
print(first_image)
img = Image.open(image_path)


# Image Resizing
resized_img = img.resize((224,224))
print(resized_img.size)
plt.imshow(resized_img)
plt.show()


# image normization
# image normalization

img_array = np.array(resized_img)

print(img_array.shape)

normalized_img = img_array / 255.0

print(normalized_img.min())
print(normalized_img.max())
