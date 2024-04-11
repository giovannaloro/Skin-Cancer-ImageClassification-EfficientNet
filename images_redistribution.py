import pandas as pd
import os 
import shutil

"This script places the dataset's images in subdirectorios based on their classes "

#open a csv files in which classes label of images are stored
os.chdir("Skin Cancer/Skin Cancer")
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
    image_class = str(image_row.dx[1])
    print(image_class)
    shutil.move(os.path.join(source_dir, image+".jpg"), os.path.join(source_dir, image_class))
