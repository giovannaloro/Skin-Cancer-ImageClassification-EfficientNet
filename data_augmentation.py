from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from os.path import join 
import os


# define parameters for data augmentation
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

# path of the image directory 
data_dir = join("dataset", "train")
classes = os.listdir(data_dir)

# path for image  save
save_to_dir = join("dataset","aug_train")

#image augmentation. for each image 5 augmented images are created
for class_name in classes:
    class_dir = os.path.join(data_dir, class_name)
    save_dir = os.path.join(save_to_dir, class_name)
    os.makedirs(save_dir, exist_ok=True)
    
    for img_file in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_file)
        img = load_img(img_path)  # Carica l'immagine
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        i = 0
        for batch in datagen.flow(x, batch_size=1, save_to_dir=save_dir, save_prefix='aug', save_format='jpeg'):
            i += 1
            if i >= 5: 
                break
