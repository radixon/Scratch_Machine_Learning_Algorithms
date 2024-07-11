import numpy as np
import pandas as pd
from collections import Counter
from sklearn import neighbors
from sklearn.metrics import confusion_matrix, classification_report

class KNN:
    def __init__(self, k):
        """
        k is the number of nearest neighbors that determines the class of a point
        """
        self.k = k
    
    def fit(self, X_train, y_train):
        """
        This function makes training data available to other functions
        """
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        """
        Calculate Euclidean Distance.

        Use Euclidean Distance to identify K-Nearest Neighbors.

        Use training data classes of the identified K-Nearest Neighbors to classify new data.
        """
        n = np.shape(X_test)[0] 
        neighbor =['']*(n)
        
        for point in range(n):
            # Euclidean Distance Calculation
            distances = np.sqrt(np.sum((self.X_train - X_test.iloc[point])**2, axis = 1))

            # Nearest Neighbors
            indices = np.argsort(distances, axis=0)

            # Classes of Nearest Neighbors
            classes = np.unique(self.y_train[indices[:self.k]])
            
            if len(classes) == 1:
                neighbor[point] = np.unique(classes)[0]
            else:
                class_count = Counter(classes)
                for i in range(self.k):
                    class_count[self.y_train[indices[i]]] += 1
                neighbor[point] = class_count.most_common(1)[0][0]
        return neighbor


if __name__ == '__main__':
    df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Smarket.csv', index_col = 0, parse_dates = True)
    print(df.head(), '\n')

    X_train = df[:'2004'][['Lag1','Lag2']]
    y_train = df[:'2004']['Direction']

    X_test = df['2005':][['Lag1','Lag2']]
    y_test = df['2005':]['Direction']

    print('\n')
    print("---Sci-Kit Learn KNN---",'\n')
    knn = neighbors.KNeighborsClassifier(n_neighbors=3)
    pred = knn.fit(X_train, y_train).predict(X_test)
    print(pred,'\n')
    print("Confusion Matrix ", '\n', confusion_matrix(y_test, pred).T, '\n')
    print(classification_report(y_test, pred, digits=3))


    print('\n')
    print("---User Defined KNN---",'\n')
    knn = KNN(k = 3)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    print(pred,'\n')
    print("Confusion Matrix ", '\n', confusion_matrix(y_test, pred).T, '\n')
    print(classification_report(y_test, pred, digits=3))
