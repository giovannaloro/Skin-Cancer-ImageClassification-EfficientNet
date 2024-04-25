import tensorflow as tf
import keras
from sklearn.metrics import confusion_matrix
from keras.applications import EfficientNetV2B0
from keras.layers import GlobalMaxPooling2D, Dense, Dropout, BatchNormalization
from keras.models import Model
import matplotlib.pyplot as plt
import numpy as np

#load and prepare dataset
batch_size = 32
img_height = 224
img_width = 224

train = keras.utils.image_dataset_from_directory(
    directory = "dataset/train",
    labels = "inferred",
    batch_size = batch_size,
    label_mode = "categorical",
    image_size = (img_width, img_height)
    )

validation = keras.utils.image_dataset_from_directory(
    directory = "dataset/validation",
    labels = "inferred",
    batch_size = batch_size,
    label_mode = "categorical",
    image_size = (img_width, img_height)
    )

test = keras.utils.image_dataset_from_directory(
    directory = "dataset/test",
    labels = "inferred",
    batch_size = batch_size,
    label_mode = "categorical",
    image_size = (img_width, img_height)
    )

#load efficient net
effnet_pretrained= EfficientNetV2B0(weights="imagenet", include_top = False, input_shape=(224,224,3))

#make all layers not trainable
effnet_pretrained.trainable = False

#add classification layers to the model
output = effnet_pretrained.layers[-1].output
x = GlobalMaxPooling2D()(output)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
classification = Dense(7, activation = "softmax")(x)
effnet = Model(effnet_pretrained.input, classification)
effnet.summary()

#compile the model and train last layers
callback = keras.callbacks.EarlyStopping(monitor='loss',patience=1, restore_best_weights = True )
effnet.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["F1Score","accuracy"] )
training = effnet.fit(train, epochs = 5, validation_data=validation, callbacks = callback)

#plot the loss
plt.plot(training.history['loss'], label='Training Loss')
plt.plot(training.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

#make 3/4 of layers trainable
effnet.trainable = True
for layer in effnet.layers:
    layer.trainable = False
for layer in effnet.layers[-270:]:
  if not isinstance(layer, BatchNormalization):
    layer.trainable = True

#train 3/4 of layers
effnet.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=["F1Score","accuracy"] )
callback = keras.callbacks.EarlyStopping(monitor='loss',patience=2, restore_best_weights = True )
training = effnet.fit(train, epochs = 25, validation_data=validation, callbacks = callback)

#plot the loss
plt.plot(training.history['loss'], label='Training Loss')
plt.plot(training.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

#make all layers trainable
effnet.trainable = True
for layer in effnet.layers:
    layer.trainable = False
for layer in effnet.layers:
  if not isinstance(layer, BatchNormalization):
    layer.trainable = True

#train 3/4 of layers
effnet.compile(optimizer=keras.optimizers.Adam(learning_rate=0.00001), loss="categorical_crossentropy", metrics=["F1Score","accuracy"] )
callback = keras.callbacks.EarlyStopping(monitor='loss',patience=2, restore_best_weights = True )
training = effnet.fit(train, epochs = 5, validation_data=validation, callbacks = callback)

#save model
effnet.save("effnet_ham10000.keras")

#plot the loss
plt.plot(training.history['loss'], label='Training Loss')
plt.plot(training.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# evaluate the model on the test set
test_loss, test_f1_score, test_accuracy = effnet.evaluate(test)

# obtain model prediction on the test set
test_predictions = effnet.predict(test)

# convert prediction in classe
predicted_classes = np.argmax(test_predictions, axis=1)

# obtain ground truth labels
true_classes = np.concatenate([y for x, y in test], axis=0)
true_classes = np.argmax(true_classes, axis=1)

# compute the confusion matrix
conf_matrix = confusion_matrix(true_classes, predicted_classes)

# show the confusion matrix
plt.figure(figsize=(10, 8))
plt.imshow(conf_matrix, cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.xticks(np.arange(7))
plt.yticks(np.arange(7))
for i in range(7):
    for j in range(7):
        plt.text(j, i, conf_matrix[i, j], ha='center', va='center', color='red')
plt.show()



