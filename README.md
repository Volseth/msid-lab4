# msid-lab4

## Introduction
Fashion-MNIST is dataset of Zalando's images representing 60,000 examples and 10,000 test set images of clothings. Each exapmle is 28px x 28px grayscale with associated label from 10 classes. Each training and test example is associated with the following labels:

| Label | Description |
| --- | --- |
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

The task is to make algorithm predict to which class given picture is associated.
## Methods
For prediction I used two models: KNN classifier and 2-Layer Convolutional neural network with 1,660,522 total parameters. As a metrics of model quality I chose accuracy metric.

### k-NN Model:
For the purpose of the task 2 I implemented k-nearest neighbours model. K-nn is the discriminative model, parameters of model are training data. k-NN calculate conditional probability distribution for each type of clothes based on given image. Classification, to which class the given picture is associated to is based on highest probability. I tested model for values of hyperparamether k, choosing one with the lowest classification error. Input data for a model is a vector, each position gives information about pixel color in grayscale.


### CNN Model:
My model is based on method of image analysis containing hand-written digits:
https://nextjournal.com/gkoehler/digit-recognition-with-keras

To create neural network I used libraries: tensorflow, keras. Structure of neural network is shown on the image below:

![Screenshot](https://github.com/Volseth/msid-lab4/blob/master/plots/network.PNG)

Neural network contains some layers, output of one layer is an input of the next layer. On the output of the neural network there is a softmax fuction mapping images to associated class. Tasks of used layers:

keras.layers.Flatten - Layer flatten input to a vector of features. In this case it's vector of length 784. (28 x 28px)

keras.layers.Conv2D - Convolutional layer with number of parameters layer can learn and window size of image analysis (kernel_size).

keras.layers.MaxPooling2D - Reducing spatial dimensions between Conv2D layers to reduce number of parameters and computing complexity. 

keras.layers.Dense - Fully connected layer with n trainable parameters.

keras.layers.BatchNormalization - Normalization of input parameters.

keras.layers.Dropout - Probability of setting input to 0 to prevent overfitting of model during training.

To increase the number of test images I used an additional function, that generates new images based on existing ones and input parameters to the generator. In this case there are horizontal-flip, rotation and width, height shift. For a model selection I extracted 10,000 images as validation set for a model. I made normalization on pixel color values to range 0-1.0 ( higher value means pixel is brighter). My model contains:
```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 25, 25, 32)        544       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 12, 12, 32)        0         
_________________________________________________________________
dropout (Dropout)            (None, 12, 12, 32)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 10, 10, 64)        18496     
_________________________________________________________________
dropout_1 (Dropout)          (None, 10, 10, 64)        0         
_________________________________________________________________
batch_normalization (BatchNo (None, 10, 10, 64)        256       
_________________________________________________________________
flatten (Flatten)            (None, 6400)              0         
_________________________________________________________________
dense (Dense)                (None, 256)               1638656   
_________________________________________________________________
dense_1 (Dense)              (None, 10)                2570      
=================================================================
Total params: 1,660,522
Trainable params: 1,660,394
Non-trainable params: 128
```
## Results
### KNN model result of training: 50,000 samples of training data, 10,000 validation data:
```
KNN TEST SET: loss: 0.2556, score: 0.7331, best_k: 25
```
![Screenshot](https://github.com/Volseth/msid-lab4/blob/master/plots/knn.png)


### CNN Model result for 30 epochs of training - 110,000 samples of training data, 10,000 validation data:
```
Train on 110000 samples, validate on 10000 samples
Epoch 1/30
110000/110000 - 75s - loss: 0.6537 - accuracy: 0.7570 - val_loss: 0.3428 - val_accuracy: 0.8676
Epoch 2/30
110000/110000 - 77s - loss: 0.4620 - accuracy: 0.8245 - val_loss: 0.2966 - val_accuracy: 0.8860
Epoch 3/30
110000/110000 - 76s - loss: 0.4168 - accuracy: 0.8423 - val_loss: 0.3433 - val_accuracy: 0.8661
Epoch 4/30
110000/110000 - 76s - loss: 0.3840 - accuracy: 0.8548 - val_loss: 0.2636 - val_accuracy: 0.9070
Epoch 5/30
110000/110000 - 76s - loss: 0.3554 - accuracy: 0.8659 - val_loss: 0.2536 - val_accuracy: 0.9059
Epoch 6/30
110000/110000 - 75s - loss: 0.3344 - accuracy: 0.8736 - val_loss: 0.2518 - val_accuracy: 0.9079
Epoch 7/30
110000/110000 - 77s - loss: 0.3206 - accuracy: 0.8777 - val_loss: 0.2531 - val_accuracy: 0.9096
Epoch 8/30
110000/110000 - 73s - loss: 0.3055 - accuracy: 0.8855 - val_loss: 0.2499 - val_accuracy: 0.9121
Epoch 9/30
110000/110000 - 74s - loss: 0.2933 - accuracy: 0.8897 - val_loss: 0.2508 - val_accuracy: 0.9111
Epoch 10/30
110000/110000 - 77s - loss: 0.2806 - accuracy: 0.8936 - val_loss: 0.2608 - val_accuracy: 0.9116
Epoch 11/30
110000/110000 - 78s - loss: 0.2734 - accuracy: 0.8974 - val_loss: 0.2423 - val_accuracy: 0.9137
Epoch 12/30
110000/110000 - 73s - loss: 0.2599 - accuracy: 0.9014 - val_loss: 0.2465 - val_accuracy: 0.9166
Epoch 13/30
110000/110000 - 72s - loss: 0.2525 - accuracy: 0.9042 - val_loss: 0.2712 - val_accuracy: 0.9070
Epoch 14/30
110000/110000 - 76s - loss: 0.2452 - accuracy: 0.9063 - val_loss: 0.2733 - val_accuracy: 0.9092
Epoch 15/30
110000/110000 - 76s - loss: 0.2377 - accuracy: 0.9096 - val_loss: 0.2621 - val_accuracy: 0.9153
Epoch 16/30
110000/110000 - 75s - loss: 0.2314 - accuracy: 0.9120 - val_loss: 0.2499 - val_accuracy: 0.9182
Epoch 17/30
110000/110000 - 73s - loss: 0.2278 - accuracy: 0.9138 - val_loss: 0.2630 - val_accuracy: 0.9143
Epoch 18/30
110000/110000 - 70s - loss: 0.2211 - accuracy: 0.9167 - val_loss: 0.2578 - val_accuracy: 0.9187
Epoch 19/30
110000/110000 - 73s - loss: 0.2114 - accuracy: 0.9208 - val_loss: 0.2614 - val_accuracy: 0.9173
Epoch 20/30
110000/110000 - 75s - loss: 0.2069 - accuracy: 0.9217 - val_loss: 0.2640 - val_accuracy: 0.9180
Epoch 21/30
110000/110000 - 78s - loss: 0.2023 - accuracy: 0.9239 - val_loss: 0.2648 - val_accuracy: 0.9177
Epoch 22/30
110000/110000 - 79s - loss: 0.1975 - accuracy: 0.9252 - val_loss: 0.2703 - val_accuracy: 0.9192
Epoch 23/30
110000/110000 - 81s - loss: 0.1939 - accuracy: 0.9257 - val_loss: 0.2845 - val_accuracy: 0.9195
Epoch 24/30
110000/110000 - 79s - loss: 0.1906 - accuracy: 0.9278 - val_loss: 0.2791 - val_accuracy: 0.9167
Epoch 25/30
110000/110000 - 79s - loss: 0.1841 - accuracy: 0.9308 - val_loss: 0.2825 - val_accuracy: 0.9205
Epoch 26/30
110000/110000 - 80s - loss: 0.1811 - accuracy: 0.9326 - val_loss: 0.2858 - val_accuracy: 0.9190
Epoch 27/30
110000/110000 - 81s - loss: 0.1749 - accuracy: 0.9338 - val_loss: 0.2692 - val_accuracy: 0.9232
Epoch 28/30
110000/110000 - 78s - loss: 0.1749 - accuracy: 0.9348 - val_loss: 0.2872 - val_accuracy: 0.9188
Epoch 29/30
110000/110000 - 79s - loss: 0.1696 - accuracy: 0.9360 - val_loss: 0.2892 - val_accuracy: 0.9198
Epoch 30/30
110000/110000 - 81s - loss: 0.1654 - accuracy: 0.9381 - val_loss: 0.2775 - val_accuracy: 0.9201
10000/10000 - 2s - loss: 0.3122 - accuracy: 0.9168
```
![Screenshot](https://github.com/Volseth/msid-lab4/blob/master/plots/neuralNetwork.png)


### Result comprasion between used methods and benchmarks:
  | Method | Results | Benchmark
| --- | --- | --- |
| KNN | k:25 -> accuracy: 0.7331 | KNeighborsClassifier	{"n_neighbors":1,"p":2,"weights":"distance"} ->	0.847    
| CNN | epochs: 30 -> accuracy: 0.9168 | 2 Conv+pooling	Preprocessing:None ->	0.876

## Usage 
### Getting the data
Keras library have included Fashion-MNIST as a built-in dataset. You don't have to download the data. 
### Running algorithms
Make sure you have installed tensorflow library for model selection and matplotlib for plots. Once you have installed libraries simply run main.py script in python interpreter.
