import numpy as np
from abc import ABC, abstractmethod

class BaseLinearRegression(ABC):
    """
    Abstract base class for linear regression models. This class provides
    the shared fit/predict logic.

    Parameter estimation is delegated via fit_
    """
    def __init__(self):
        self.intercept_ = None
        self.coef_ = None
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Fit the model to training data.

        Args:
            X_train (np.ndarray): Predictor matrix
            y_train (np.ndarray): Response vector

        Returns:
            self: This allows calls to be chained together.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.rows, self.cols = X_train.shape

        coefficients = np.concatenate((np.ones((self.rows, 1)), X_train), axis=1)

        params = self.fit_(coefficients, y_train)
        self.intercept_ = params[0]
        self.coef_ = params[1:]
        return self

    def predict(self, X_test: np.ndarray) ->  np.ndarray:
        """
        Predict target values for X_test

        Args:
            X_test (np.ndarray): Test matrix
        
        Returns:
            np.ndarray: Predicted values
        """
        target = np.dot(X_test, self.coef_) + self.intercept_
        return target
    
    @abstractmethod
    def fit_(self, X_bias: np.ndarray, y: np.ndarray):
        """
        This method returns the parameters after fitting the data to the model.
        """
        pass