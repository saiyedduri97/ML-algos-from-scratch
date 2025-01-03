import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as optimize
import seaborn as sns
from synthetic_data_preparation import preparing_data


class KernelSVM:
    def __init__(self, kernel=None, reg_param=0.01, **kernel_params):
        """
        Initialize the Kernel SVM.

        Parameters:
            kernel (callable): Kernel function (default: linear_kernel).
            reg_param (float): Regularization parameter (default: 0.01).
            **kernel_params: Additional parameters for the kernel function.
        """
        self.kernel = kernel if kernel is not None else self.linear_kernel
        self.reg_param = reg_param
        self.kernel_params = kernel_params
        self.alpha_optimized = None
        self.kernel_matrix = None

    @staticmethod
    def linear_kernel(x1, x2):
        """
        Compute the linear mapping of two vectors.

        Parameters:
            x1, x2 (numpy array): Input vectors.

        Returns:
            float: Dot product of x1 and x2.
        """
        return np.dot(x1, x2)

    @staticmethod
    def polynomial_kernel(x1, x2, gamma=1, const=1, degree=3):
        """
        Compute the polynomial features between two vectors.

        Parameters:
            x1, x2 (numpy array): Input vectors.
            gamma (float): Scaling parameter for the polynomial kernel.
            const (float): Constant term.
            degree (int): Degree of the polynomial.

        Returns:
            float: Polynomial kernel value.
        """
        return (gamma * np.dot(x1, x2) + const) ** degree

    @staticmethod
    def gaussian_kernel(x1, x2, gamma=0.5):
        """
        Compute the Gaussian (RBF) kernel.

        Parameters:
            x1, x2 (numpy array): Input vectors.
            gamma (float): Parameter defining the width of the Gaussian.

        Returns:
            float: Gaussian kernel value.
        """
        return np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)

    def kernel_mat(self, X):
        """
        Compute the kernel matrix for the dataset X.

        Parameters:
            X (numpy array): Input feature matrix of shape (n_samples, n_features).

        Returns:
            numpy array: Kernel matrix of shape (n_samples, n_samples).
        """
        rows = X.shape[0]
        kernel_matrix = np.zeros((rows, rows))
        for idx1 in range(rows):
            for idx2 in range(rows):
                kernel_matrix[idx1, idx2] = self.kernel(X[idx1], X[idx2], **self.kernel_params)
        return kernel_matrix

    def hinge_losses(self, X, y, alpha=None, b=0):
        """
        Compute hinge loss for the dataset X.

        Parameters:
            X (numpy array): Input feature matrix of shape (n_samples, n_features).
            y (numpy array): Target labels of shape (n_samples,).
            alpha (numpy array): Dual coefficients of shape (n_samples,). Defaults to ones.
            b (float): Bias term. Defaults to 0.

        Returns:
            numpy array: Hinge loss values for each sample.
        """
        if alpha is None:
            alpha = np.ones(X.shape[0])

        kernel_matrix = self.kernel_mat(X)
        z = np.dot(kernel_matrix, alpha) + b
        z = z.reshape(y.shape)

        return np.maximum(0, 1 - y * z)

    def regularized_loss(self, alpha, X, y, b=0):
        """
        Compute the regularized loss.

        Parameters:
            alpha (numpy array): Dual coefficients of shape (n_samples,).
            X (numpy array): Input feature matrix of shape (n_samples, n_features).
            y (numpy array): Target labels of shape (n_samples,).
            b (float): Bias term. Defaults to 0.

        Returns:
            float: Regularized loss value.
        """
        hinge_loss_vals = self.hinge_losses(X, y, alpha, b)
        reg_term = self.reg_param * (alpha @ self.kernel_matrix @ alpha.T)
        return np.mean(hinge_loss_vals) + reg_term

    def optimize_alpha(self, X, y, b=0):
        """
        Optimize the dual coefficients (alpha) using simplex optimization.

        Parameters:
            X (numpy array): Input feature matrix of shape (n_samples, n_features).
            y (numpy array): Target labels of shape (n_samples,).
            b (float): Bias term. Defaults to 0.

        Returns:
            numpy array: Optimized dual coefficients (alpha).
        """
        self.kernel_matrix = self.kernel_mat(X)
        alpha_initial = np.ones(X.shape[0])

        self.alpha_optimized = optimize.fmin(
            func=self.regularized_loss,
            x0=alpha_initial,
            args=(X, y, b),
            disp=False
        ).reshape(1, -1)

        return self.alpha_optimized

    def predict(self, X, y, x_new, b=0):
        """
        Predict the class labels for new data points.

        Parameters:
            X (numpy array): Training data of shape (n_samples, n_features).
            y (numpy array): Training labels of shape (n_samples,).
            x_new (numpy array): Test data of shape (n_test_samples, n_features).
            b (float): Bias term. Defaults to 0.

        Returns:
            numpy array: Predicted class labels for each test point.
        """
        predictions = []
        for x in x_new:
            decision_value = 0
            for i in range(len(X)):
                decision_value += self.alpha_optimized[0][i] * y[i] * self.kernel(X[i], x, **self.kernel_params)
            decision_value += b
            predictions.append(np.sign(decision_value))
        return np.array(predictions)

if __name__=="__main__":
    # Generating synthetic data
    X,y,probabilities=preparing_data(w_true=np.array([1, 2]),b=0,n_samples=25, feature_range=(-3, 3),random_seed=42)

    #Initialize svm
    linear_svm = KernelSVM(kernel=KernelSVM.linear_kernel, reg_param=0.001)

    # Optimize coefficient of support vectors(alpha):
    alpha_optimized=linear_svm.optimize_alpha(X,y,b=0)
    print("Optimized alpha:", alpha_optimized)

    # Predict new data
    x_new = np.array([[1, 2], [3, 4]])
    predictions = linear_svm.predict(X, y, x_new, b=0)
    print("Predicted labels:", predictions)


    
