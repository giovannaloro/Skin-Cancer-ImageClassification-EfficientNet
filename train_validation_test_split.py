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
#define number of images to move i test directory for each class
N = 30

#create the test subdirectory
os.mkdir("test")
for image_class in image_classes:
    os.mkdir(f"test/{image_class}")
    
#move 30 random selected images from each class directory to the test directory
for image_class in image_classes:
    images = list(os.listdir(image_class))
    for i in range(N):
        image = images.pop(random.randrange(len(images)))
        shutil.move(join(image_class,image), join("test",image_class,image))

#create the train subdirectory 
os.mkdir("train")
for image_class in image_classes:
        os.mkdir(f"train/{image_class}")

#move 80% of images to the train directory for each class
for image_class in image_classes:
     images = list(os.listdir(image_class))
     for i in range((len(images)//(10))*8):
        image = images.pop(random.randrange(len(images)))
        shutil.move(join(image_class,image), join("train",image_class,image))

#create the validation subdirectory
os.mkdir("validation")
for image_class in image_classes:
    os.mkdir(f"validation/{image_class}")

#move all remaining images in the validation directory 
for image_class in image_classes:
     images = list(os.listdir(image_class))
     while(len(images) > 0):
        image = images.pop(len(images)-1)
        shutil.move(join(image_class,image), join("validation",image_class,image))
