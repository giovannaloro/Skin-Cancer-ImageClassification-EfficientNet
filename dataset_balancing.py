import pandas as pd
import os 
import shutil
import random
from os.path import join

"This script places the dataset's images in subdirectorios based on their classes "
#move in the dataset directory
os.chdir("dataset")
#define all classes
image_classes = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
#move all augmented images in the train set
for image_class in image_classes:
    images = list(os.listdir(join("aug_train",image_class)))
    while(len(images)>0):
        image = images.pop(random.randrange(len(images)))
        shutil.move(join("aug_train",image_class,image), join("train",image_class,image))
        
#count the number of images for each class and take the minimum
image_counts = []
for image_class in image_classes:
    image_count = len(list(os.listdir(join("train",image_class))))
    print(image_count)
    image_counts.append(image_count)
m = min(image_counts)

#select randomly m images for each class and delete all non selected images
for image_class in image_classes:
    images = list(os.listdir(join("train",image_class)))
    while(len(images)>m):
        image = images.pop(random.randrange(len(images)))
        os.remove(join("train",image_class,image))
