
import pandas as pd
import os 
import shutil
import random
from os.path import join
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

#open a csv files in which classes label of images are stored
os.chdir("dataset")
df = pd.read_csv("HAM10000_metadata.csv")

#create a list of images names
images_list = df['image_id'].to_list()

#create a list of possible classes and remove duplicates
classes = df['dx'].to_list()
classes = list(set(classes))

#create a directory for each class
for clas in classes:
    os.mkdir(clas)

#move each image in his class directory
source_dir = os.getcwd()
for image in images_list:
    image_row = df.loc[df['image_id'] == image]
    clas = image_row.iloc[0]['dx']
    shutil.move(os.path.join(source_dir, image+".jpg"), os.path.join(source_dir, clas))

os.remove("HAM10000_metadata.csv")

#define all classes
classes

#define number of images to move in test directory for each class
N = 30

#create the test subdirectory
os.mkdir("test")
for clas in classes:
    os.mkdir(f"test/{clas}")
    
#move 30 random selected images from each class directory to the test directory
for clas in classes:
    images = list(os.listdir(clas))
    for i in range(N):
        image = images.pop(random.randrange(len(images)))
        shutil.move(join(clas,image), join("test",clas,image))

#create the train subdirectory 
os.mkdir("train")
for clas in classes:
        os.mkdir(f"train/{clas}")

#move 80% of images to the train directory for each class
for clas in classes:
     images = list(os.listdir(clas))
     for i in range((len(images)//(10))*8):
        image = images.pop(random.randrange(len(images)))
        shutil.move(join(clas,image), join("train",clas,image))

#create the validation subdirectory
os.mkdir("validation")
for clas in classes:
    os.mkdir(f"validation/{clas}")

#move all remaining images in the validation directory 
for clas in classes:
     images = list(os.listdir(clas))
     while(len(images) > 0):
        image = images.pop(len(images)-1)
        shutil.move(join(clas,image), join("validation",clas,image))


# define parameters for data augmentation
datagen = ImageDataGenerator(
    rotation_range=90,
    width_shift_range=0,
    height_shift_range=0,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

# path of the image directory 
data_dir = "train"

# path for image  save
save_to_dir = "aug_train"

#image augmentation. for each image 10 augmented images are created
for class_name in classes:
    class_dir = os.path.join(data_dir, class_name)
    save_dir = os.path.join(save_to_dir, class_name)
    os.makedirs(save_dir, exist_ok=True)
    image_num = len(list(os.listdir(class_dir)))
    augmentation_factor  = (1000-image_num)//image_num + 1
    for img_file in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_file)
        img = load_img(img_path)  # Carica l'immagine
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        i = 0
        for batch in datagen.flow(x, batch_size=1, save_to_dir=save_dir, save_prefix='aug', save_format='jpeg'):
            i += 1
            if i >= augmentation_factor: 
                break

#move all augmented images in the train set
for clas in classes:
    images = list(os.listdir(join("aug_train",clas)))
    while(len(images)>0):
        image = images.pop(random.randrange(len(images)))
        shutil.move(join("aug_train",clas,image), join("train",clas,image))
        
#count the number of images for each class and take the minimum
image_counts = []
for clas in classes:
    image_count = len(list(os.listdir(join("train",clas))))
    print(image_count)
    image_counts.append(image_count)
m = min(image_counts)

#select randomly m images for each class and delete all non selected images
for clas in classes:
    images = list(os.listdir(join("train",clas)))
    while(len(images)>m):
        image = images.pop(random.randrange(len(images)))
        os.remove(join("train",clas,image))
