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

The task is to make algorithm predicting to which class is asocciated given picture.
## Methods
For prediction i used two models: KNN classifier and 3-Layer Convolutional neural network with 2,750,730 trainable parameters. As a metrics of model quality i chose accuracy metric.

### k-NN Model:
For the purpouse of the task 2 I implemented k-nearest neighbours model. K-nn is discriminative model, parameters of model are training data. k-NN calculate conditional probability distribution for each type of clothes based on given image. Classification, to which class associated is the image is to choose the class which probability is highest. I tested model for values of hyperparamether k, choosing one with the lowest classification error. Input data is for a model is a vector, each position gives information about pixel color in grayscale.


### CNN Model:
My model is based on method of image analysis containing hand-written digits:
https://nextjournal.com/gkoehler/digit-recognition-with-keras

To create neural network i used libraries: tensorflow, keras. Structure of neural network is shown on the image below:

Neural network contains some layers, output of one layer is an input of the next layer. On the output of the neural network there is a softmax fuction mapping images to associated class. Tasks of used layers:

keras.layers.Flatten - Layer flatten input to a vector of features. In this case it's vector of length 784. (28 x 28px)

keras.layers.Conv2D - Convolutional layer with number of parameters layer can learn and window size of learning (kernel_size).

keras.layers.MaxPooling2D - Reducing spatial dimensions between Conv2D layers. 

keras.layers.Dense - Fully connected layer with n trainable parameters.

keras.layers.BatchNormalization - Normalization of input parameters.

keras.layers.Dropout - Probability of setting input to 0 to prevent overfitting of model during training.

To increase the number of test images i used an additional function, that generates new images based on existing ones and input parameters of generator. In this case there are horizontal-flip, rotation, oraz width, height shift. For a model selection i extracted 10,000 images as validation set for a model. I made normalization on pixel color values to range 0-1.0 ( higher value means pixel is darker). My model contains:
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
### KNN model result of trainig: 50,000 samples of training data, 10,000 validation data:
```
KNN TEST SET: loss: 0.267, score: 0.733, best_k: 25
```
### CNN Model result for 15 epochs of training - 110,000 samples of training data, 10,000 validation data:
```
Train on 110000 samples, validate on 10000 samples
Epoch 1/15
110000/110000 - 69s - loss: 0.6531 - accuracy: 0.7570 - val_loss: 0.3849 - val_accuracy: 0.8586
Epoch 2/15
110000/110000 - 70s - loss: 0.4650 - accuracy: 0.8244 - val_loss: 0.2998 - val_accuracy: 0.8900
Epoch 3/15
110000/110000 - 69s - loss: 0.4176 - accuracy: 0.8418 - val_loss: 0.2815 - val_accuracy: 0.8958
Epoch 4/15
110000/110000 - 69s - loss: 0.3820 - accuracy: 0.8551 - val_loss: 0.2659 - val_accuracy: 0.9003
Epoch 5/15
110000/110000 - 73s - loss: 0.3553 - accuracy: 0.8654 - val_loss: 0.2517 - val_accuracy: 0.9058
Epoch 6/15
110000/110000 - 69s - loss: 0.3381 - accuracy: 0.8714 - val_loss: 0.2583 - val_accuracy: 0.9049
Epoch 7/15
110000/110000 - 69s - loss: 0.3192 - accuracy: 0.8788 - val_loss: 0.2455 - val_accuracy: 0.9138
Epoch 8/15
110000/110000 - 69s - loss: 0.3062 - accuracy: 0.8842 - val_loss: 0.2537 - val_accuracy: 0.9090
Epoch 9/15
110000/110000 - 70s - loss: 0.2942 - accuracy: 0.8887 - val_loss: 0.2451 - val_accuracy: 0.9151
Epoch 10/15
110000/110000 - 68s - loss: 0.2846 - accuracy: 0.8920 - val_loss: 0.2335 - val_accuracy: 0.9157
Epoch 11/15
110000/110000 - 71s - loss: 0.2700 - accuracy: 0.8971 - val_loss: 0.2592 - val_accuracy: 0.9132
Epoch 12/15
110000/110000 - 69s - loss: 0.2603 - accuracy: 0.9014 - val_loss: 0.2505 - val_accuracy: 0.9157
Epoch 13/15
110000/110000 - 73s - loss: 0.2492 - accuracy: 0.9054 - val_loss: 0.2550 - val_accuracy: 0.9187
Epoch 14/15
110000/110000 - 71s - loss: 0.2428 - accuracy: 0.9073 - val_loss: 0.2553 - val_accuracy: 0.9173
Epoch 15/15
110000/110000 - 70s - loss: 0.2356 - accuracy: 0.9102 - val_loss: 0.2292 - val_accuracy: 0.9214
10000/10000 - 1s - loss: 0.2654 - accuracy: 0.9144
```

### Result comprasion between used methods and benchmarks:
  | Method | Results | Benchmark
| --- | --- | --- |
| KNN | k:25 -> accuracy: 0.733 | KNeighborsClassifier	{"n_neighbors":1,"p":2,"weights":"distance"} ->	0.847    
| CNN | epochs: 15 -> accuracy: 0.9144 | 2 Conv+pooling	Preprocessing:None ->	0.876

## Usage 
### Getting the data
Keras library have included Fashion-MNIST as a built-in dataset. You don't have to download the data. 
### Running algorithms
Make sure you have installed tensorflow library for model selection and matplotlib for plots. Once you have installed libraries simply run main.py script in python interpreter.
