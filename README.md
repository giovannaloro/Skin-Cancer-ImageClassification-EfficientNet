# Overview

This repository contains code for performing image classification on the HAM10000 dataset using EfficientNetV2B0 architecture. The HAM10000 dataset is a collection of dermatoscopic images of common pigmented skin lesions. EfficientNetV2B0 is a convolutional neural network architecture that has demonstrated state-of-the-art performance on various image classification tasks.

# Dataset

The HAM10000 dataset consists of 10,000 dermatoscopic images which are labeled into 7 different classes: melanoma, melanocytic nevus, basal cell carcinoma, actinic keratosis, benign keratosis (solar lentigo / seborrheic keratosis / lichen planus-like keratosis), dermatofibroma, and vascular lesion.

# EfficientNetV2B0

EfficientNetV2B0 is a variant of the EfficientNet architecture, which employs a novel compound scaling method to uniformly scale all dimensions of depth/width/resolution using a fixed set of scaling coefficients. This results in models that are both smaller and more accurate than previous models.

# Methodology

* Data Augmentation: To improve model generalization and robustness, data augmentation techniques such as rotation, horizontal flip, and zoom were applied to the training dataset.
* Classes Rebalance: Since the dataset is imbalanced, with some classes having significantly fewer samples than others, rebalancing techniques were employed to ensure fair representation of all classes during training.

# Training Process
The model was trained by doing fine-tuning and transfer-learning starting from Imagenetweights
The training process consisted of three steps:

* Fc layers training (5 epochs): During this phase, only the fully connected (FC) layers of the EfficientNetV2B0 model were trained while keeping the rest of the layers frozen. 

* Transfer learning on 3/4 of the net (25 epochs): In this phase, approximately three-quarters of the layers of the EfficientNetV2B0 model, excluding batch normalization layers, were unfrozen and trained using a lower learning rate. This allowed the model to fine-tune its representations on the HAM10000 dataset.

* Full Training (5 epochs): Finally, all layers of the EfficientNetV2B0 model, except for batch normalization layers, were unfrozen and trained at a ver low learning rate for a few additional epochs. This fine-tuned the entire model to better adapt to the specifics of the dataset and improve classification performance.


# Results

After training the EfficientNetV2B0 model on the HAM10000 dataset and evaluating it on a separate test set, an F1 Score of 0.7 was achieved. This indicates the model's ability to effectively classify skin lesions into their respective categories.
