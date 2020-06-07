from knn import *
from tensor import *


def load_mnist_keras():
    fashion_mnist = keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    return train_images, test_images, train_labels, test_labels


X_train, X_test, y_train, y_test = load_mnist_keras()

X_train_knn = X_train.reshape(len(y_train), 784)
X_test_knn = X_test.reshape(len(y_test), 784)

X_val_knn = X_train_knn[0:10000]
y_val_knn = y_train[0:10000]

X_train_knn = X_train_knn[10000:60000]
y_train_knn = y_train[10000:60000]


def knn_model(trainingX, valX, trainingY, valY, testX, testY, k_values):
    error_best, best_k, errors_k = model_selection_knn(valX, trainingX, valY, trainingY, k_values)
    y_sorted = sort_train_labels_knn(hamming_distance(testX, trainingX), trainingY)
    score = classification_error(p_y_x_knn(y_sorted, best_k), testY)
    return score, error_best, best_k, errors_k


def run_training():
    predictFashion(X_train, y_train, X_test, y_test)

    print("Model selection for k :[1, 3, 5, 10, 15, 20, 25, 30, 50, 100]")
    k_values = [1, 2, 3, 5, 10, 15, 20, 25, 30, 50, 100]
    score, error_best, best_k, errors_k = knn_model(X_train_knn, X_val_knn, y_train_knn, y_val_knn, X_test_knn, y_test,
                                                    k_values)
    scores = 1.0 - np.array(errors_k)

    plt.xlabel('Neighbours k')
    plt.ylabel('Accuracy')
    plt.title("Model selection for KNN")
    plt.plot(k_values, scores, 'r-', color='#FFCC55')
    plt.draw()
    plt.show()

    messageKNN = f"KNN TEST SET: loss: {error_best}, score: {1 - score}, best_k: {best_k}"
    print(messageKNN)


run_training()
