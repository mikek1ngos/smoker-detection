# smoker-detection

This project consists of a smoker detection program in which ,with the help of NVIDIA ,an artificial intelligence recognizes patterns in order to accurately determinate if the image shows a smoking person or not .
I therefore chose this dataset (from kaggle) simply because i found it interesting but also because with "detectnet" you can find smokers in non-smoking area like in an airplane or a hospital ,etc.

![add image descrition here](direct image link here)

## From the start 
For reference , we first started with learning python and getting familiar with the jetson nano , a small, low-cost, and powerful computer module that's designed for artificial intelligence (AI) and machine learning (ML) .After that we got familiar with linux and machine learning : the "thumbs project "  was the first thing we did that ressembled image classification .We then moved on to creating our own model ,heart of this project .  ![thumbs](https://github.com/mikek1ngos/smoker-detection/assets/174377643/7534a8be-ea00-42cb-9231-a63c1f3d23fe)


## The Algorithm
Based on NVIDIA's course ,we were able to create an AI to which I "fed" the "smoker detection" dataset .

**In this process ,I used diverse models ,databases ,libraries ,and series of python code of which imagenet ,resnet18 ,docker and train.py ,as well as different apps like jupyterlab ,VScode ,PuTTy and more**
- ResNet-18 is a convolutional neural network (CNN) model that's 18 layers deep and is part of the ResNet (Residual Network) family. It's designed for image classification tasks, and can classify images into 1,000 object categories, such as animals, keyboards, and mice. ResNet-18 is trained on over a million images from the ImageNet database, and a pretrained version of the network is available.
- A Docker container image is a lightweight, standalone, executable package of software that includes everything needed to run an application: code, runtime, system tools, system libraries and settings.
- ImageNet is a large, publicly available database of annotated images used for computer vision tasks, such as image classification and object detection. The database contains over 14 million high-resolution images, organized into more than 20,000 categories using WordNet synonym sets. Each image is hand-annotated to indicate what objects are pictured, and at least one million images also include bounding boxes. ImageNet's creators provide URLs and thumbnails of the images, but do not own them.
- Train.py is a python code already intalled on the jetson nano  .It can be accessed through VScode and helps train our model to attain our goal .This code is composed of many lessons covered in class like conditions ,loops ,libraries ,etc. .

**In fact ,the learning process of a  model is devided in three steps  :**
- First is training : 80% of the images are stored in the train folder
- Second is validating : 10% of the images are stored in the val folder
- Third is testing : 10% of the images are stored in the test folder

**Throughout the testing and validation phases there are many indicators that showcase the success or not of your project:**
- Train Loss     , *ex* : 5.6727e-01
- Train Accuracy , *ex* : 69.9860
- Val Loss       , *ex* : 5.6227e-01
- Val Accuracy   , *ex* :  71.1111

**The testing accuracy is determined by various factors :**
-Batch size : A hyperparameter in deep learning training that refers to the number of training samples that are fed into a neural network at once. It's a crucial parameter in training Convolutional Neural Networks (CNNs) because it directly affects the efficiency and effectiveness of the training process. 
-Epochs number : The number of epochs is the number of complete iterations of the training dataset. A batch must have a minimum size of one and a maximum size that is less than or equal to the number of samples in the training dataset. For the number of epochs, it is possible to choose an integer value between one and infinity.
-Workers number : It's the number of processes that generate batches in parallel. A high enough number of workers assures that CPU computations are efficiently managed, i.e. that the bottleneck is indeed the neural network's forward and backward operations on the GPU (and not data generation).

**Neural networks**
Neural networks are computational models inspired by the human brain's structure and functioning. They consist of interconnected nodes (neurons) organized in layers, where each neuron processes and transmits information. These networks learn patterns and relationships from data through a process called training, adjusting the strength of connections (weights) between neurons to optimize performance on specific tasks such as classification or prediction.

##



  


## Running this project


[View a video explanation here](video link)
