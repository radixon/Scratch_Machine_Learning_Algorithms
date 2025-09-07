import os
import sys
import numpy as np



sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.models.base_linearregression import BaseLinearRegression

class LinearRegression(BaseLinearRegression):
    """
    Ordinary Least Squares Regression (closed form)
    """
    def fit_(self, X_bias: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Apply the Linear Regression Algorithm.
        beta_hat = (X.T • X)¯¹ • X.T • y

        Args:
            X_bias (np.ndarray): Observations matrix
            y (np.ndarray): Response vector

        Returns:
            np.ndarray: Parameter vector
        """
        return np.linalg.inv(X_bias.T @ X_bias) @ X_bias.T @ y