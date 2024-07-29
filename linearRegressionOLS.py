import numpy as np
import pandas as pd

class LinearRegressionOLS:
    def fit(self, X_train, y_train):
        """
        This method makes training data available to other methods
        """
        self.X_train = X_train
        self.y_train = y_train
        self.rows = X_train.shape[0]
        self.cols = X_train.shape[1]
        self._parameters()
    
    def predict(self,X_test):
        """
        y_hat = X * beta_hat = X * (X_transpose * X)^-1 * X_transpose * y = H * y
        where H computes the orthogonal projection
        """
        y_pred = np.dot(X_test,self.coef_) + self.intercept_
        return y_pred
    

    def _parameters(self):
        """
        This method calculates the parameters using Ordinary Least Squares
        """
        inputs = np.concatenate((np.ones((self.rows, 1)), self.X_train), axis = 1)
        beta = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(inputs), inputs)),np.transpose(inputs)), self.y_train)
        self.intercept_ = beta[0]
        self.coef_ = beta[1:]


if __name__ == '__main__':
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score

    df = pd.read_csv("../mtcars.csv", sep = ',')
    print(df.head())
    drop_list = ['model', 'mpg', 'cyl', 'disp', 'drat', 'qsec', 'vs', 'am', 'gear', 'carb']
    Predictors = df.drop(drop_list,axis=1)
    Response = df['mpg']
    X_train, X_test, y_train, y_test = train_test_split(Predictors, Response, test_size = 0.3, random_state=42)
    

    print('\n')
    print("---Sci-Kit Learn Linear Regression Model---",'\n')
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    print("Intercept: ", lr.intercept_)
    print("Coefficients: ", lr.coef_)
    pred = lr.predict(X_test)
    print("Predictions:",'\t',pred,'\n')
    print("MSE:", '\t', mean_squared_error(y_test, pred).T)
    print("R2_Score:", '\t', r2_score(y_test, pred).T, '\n')

    print('\n')
    print("---User Defined Linear Regression Model---",'\n')
    lr = LinearRegressionOLS()
    lr.fit(X_train, y_train)
    print("Intercept: ", lr.intercept_)
    print("Coefficients: ", lr.coef_)
    pred = lr.predict(X_test)
    print("Predictions:",'\t',pred,'\n')
    print("MSE:", '\t', mean_squared_error(y_test, pred).T)
    print("R2_Score:", '\t', r2_score(y_test, pred).T, '\n')

    
