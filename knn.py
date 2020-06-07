import numpy as np


def hamming_distance(X, X_train):
    """
    Zwróć odległość Hamminga dla obiektów ze zbioru *X* od obiektów z *X_train*.

    :param X: zbiór porównywanych obiektów N1xD
    :param X_train: zbiór obiektów do których porównujemy N2xD
    :return: macierz odległości pomiędzy obiektami z "X" i "X_train" N1xN2
    """
    X_train = X_train.transpose()
    return (~X).astype(int) @ X_train.astype(int) + X.astype(int) @ (~X_train).astype(int)
    pass


def sort_train_labels_knn(Dist, y):
    """
    Posortuj etykiety klas danych treningowych *y* względem prawdopodobieństw
    zawartych w macierzy *Dist*.

    :param Dist: macierz odległości pomiędzy obiektami z "X" i "X_train" N1xN2
    :param y: wektor etykiet o długości N2
    :return: macierz etykiet klas posortowana względem wartości podobieństw
        odpowiadającego wiersza macierzy Dist N1xN2

    Do sortowania użyj algorytmu mergesort.
    """
    prob = Dist.argsort(kind='mergesort')
    return y[prob]
    pass


def p_y_x_knn(y, k):
    """
    Wyznacz rozkład prawdopodobieństwa p(y|x) każdej z klas dla obiektów
    ze zbioru testowego wykorzystując klasyfikator KNN wyuczony na danych
    treningowych.

    :param y: macierz posortowanych etykiet dla danych treningowych N1xN2
    :param k: liczba najbliższych sasiadow dla KNN
    :return: macierz prawdopodobieństw p(y|x) dla obiektów z "X" N1xM
    """
    M = np.unique(y).shape[0]
    result = []
    for i in range(np.shape(y)[0]):
        result_helper = []
        for j in range(k):
            result_helper.append(y[i][j])
        line = np.bincount(result_helper, None, M)
        row_h = []
        for row in line:
            row_h.append(row / k)
        result.append(row_h)
    return result


def classification_error(p_y_x, y_true):
    """
    Wyznacz błąd klasyfikacji.

    :param p_y_x: macierz przewidywanych prawdopodobieństw - każdy wiersz
        macierzy reprezentuje rozkład p(y|x) NxM
    :param y_true: zbiór rzeczywistych etykiet klas 1xN
    :return: błąd klasyfikacji
    """
    n = len(p_y_x)
    values = len(p_y_x[0]) - 1
    res = 0
    for i in range(0, n):
        y_pred = (values - np.argmax(p_y_x[i][::-1]))
        if y_pred != y_true[i]:
            res += 1
    return res / n


def model_selection_knn(X_val, X_train, y_val, y_train, k_values):
    """
    Wylicz bład dla różnych wartości *k*. Dokonaj selekcji modelu KNN
    wyznaczając najlepszą wartość *k*, tj. taką, dla której wartość błędu jest
    najniższa.

    :param X_val: zbiór danych walidacyjnych N1xD
    :param X_train: zbiór danych treningowych N2xD
    :param y_val: etykiety klas dla danych walidacyjnych 1xN1
    :param y_train: etykiety klas dla danych treningowych 1xN2
    :param k_values: wartości parametru k, które mają zostać sprawdzone
    :return: krotka (best_error, best_k, errors), gdzie "best_error" to
        najniższy osiągnięty błąd, "best_k" to "k" dla którego błąd był
        najniższy, a "errors" - lista wartości błędów dla kolejnych
        "k" z "k_values"
    """

    sorted_hamming = sort_train_labels_knn(hamming_distance(X_val, X_train), y_train)
    errors = []
    for k in k_values:
        print(f"Model for {k}")
        error = classification_error(p_y_x_knn(sorted_hamming, k), y_val)
        errors.append(error)
    return errors[int(np.argmin(errors))], k_values[int(np.argmin(errors))], errors
