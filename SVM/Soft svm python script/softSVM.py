import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from synthetic_data_preparation import preparing_data

class SVM:
    def __init__(self, reg_param=0.01):
        self.reg_param = reg_param  # Regularization parameter
        self.w_optimized = None    # Optimized weights
        self.b_optimized = 0       # Bias term

    def hinge_losses(self, X, y, w, b):
        """
        Function: Compute hinge loss on the data X with respect to labels y=+1 and y=-1.

        Parameters:
            X(numpy array): Input feature matrix of shape (no.samples, no.features)
            y(numpy array): Target labels mapped to +1 and -1, shape (no.samples, 1)
            w(numpy array): Weight vector of shape (1, no.features)
            b(numpy array): Bias term,scalar or (no.samples, 1) if exists
        Returns:
            loss(numpy array): Returns hinge loss vectors of the shape (1, no.samples) with each corrsonding to hingeloss of datapoint in X.
                            Loss for misclassified points increases linearly as the 
        """
        w = w.reshape(1, -1)  # Ensure w is reshaped to (1, d)
        z = (X @ w.T + b)  # Compute decision function
        return np.maximum(0, 1 - y * z)  # Hinge loss

    def regularized_loss(self, w, X, y, b=None):
        """
        Function: Compute the regularized loss, which includes hinge loss and a regularization term.

        Parameters:
            X(numpy array): Input feature matrix of shape (no.samples, no.features)
            y(numpy array): Target labels mapped to +1 and -1, shape (no.samples, 1)
            w(numpy array): Weight vector of shape (1, no.features)
            b(float): Bias term  

            Returns:
            regularizedloss(float): Returns regularized loss
        """
        w = w.reshape(1, -1)  # Ensure w is reshaped to (1, d)
        hinge_loss_vals = self.hinge_losses(X, y, w, b)  # Compute hinge loss
        reg_term = self.reg_param * np.linalg.norm(w)**2  # Regularization term
        return np.mean(hinge_loss_vals) + reg_term  # Total loss
    
    def optimize_weights(self, X, y, b=0):
        """
        Function: Optimization of a regualarized loss function using Nelder-Mead Simplex function.

        Since hingeloss is not differentiable, gradient based algorithms like SGD is not possible. So we choose to optimize 
        using the funtion values using simplex algorithm.

        Parameters:
            X(numpy array): Input feature matrix of shape (no.samples, no.features)
            y(numpy array): Target labels mapped to +1 and -1, shape (no.samples,1)
            b(float): Bias intercept for the linear model
            opt_func: A regularized function is optimzed with additional parameters b and regulation parameters
        Returns:
            w_optimized: tuple(1,no.features): Returns optimal weight vector w 
        """
        # Initialize weights
        w_initial = np.zeros((1, X.shape[1]))

        # Optimize
        self.w_optimized = optimize.fmin(
            func=self.regularized_loss,
            x0=w_initial.flatten(),
            args=(X, y, b),
            disp=False  # Suppress output during optimization
        ).reshape(1, -1)
        self.b_optimized = b
        return self.w_optimized

    def predict(self, X_test):
        """
        Predict labels for test data.

        Parameters:
        X_test(numpy array):New datapoint X_test with shape of (no.samples,no.features) to be classified by the hyperplane formed by (w_optimized,b_optimized)

        Returns*(numpy array):Returns class of the new data points with shape (no.samples,1)
        """
        return np.sign(np.dot(X_test, self.w_optimized.T) + self.b_optimized)

    def contour_plot_visualization(self, X, y, w_true, step=0.03, save_fig=True):
        """
        Plot the regularized loss landscape and visualize the optimal and true weights.

        Parameters:
            X (ndarray): Feature matrix (n_samples, n_features).
            y (ndarray): Target labels (n_samples,).
            w_optimized (ndarray): Optimized weight vector.
            w_true (ndarray): True weight vector.
            regularized_loss (function): Function to compute regularized loss.
            reg_param (float): Regularization parameter.
            step (float): Step size for creating the meshgrid.
            save_fig(bool): Saves the the boundary plot in the current folder.
            
        Returns:
            Displays and saves optimization of weights plot 
        """
        # Define ranges for the meshgrid
        x_min = min(X[:, 0]) - 0.5
        x_max = max(X[:, 0]) + 0.5
        y_min = min(X[:, 1]) - 0.5
        y_max = max(X[:, 1]) + 0.5

        # Create meshgrid
        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, step),
            np.arange(y_min, y_max, step)
        )

        mesh_points = np.c_[xx.ravel(), yy.ravel()]

        # Compute regularized losses for each point in the meshgrid
        reg_losses = [
            self.regularized_loss(
                w=point.reshape(1, -1),
                X=X,
                y=y,
                b=self.b_optimized
            )
            for point in mesh_points
        ]
        reg_losses = np.array(reg_losses).reshape(xx.shape)

        # Plot the contour
        fig, ax = plt.subplots()
        CS = ax.contour(xx, yy, reg_losses, 60)
        ax.scatter(self.w_optimized[0][0], self.w_optimized[0][1], c="r", label="Optimal Weight")
        ax.scatter(w_true[0], w_true[1], marker="+", c="k", label="True Weight")
        ax.set_title(f"$ {self.reg_param:.4f} |w|^2 + R_s(w)$")
        ax.set_xlabel('$w_1$')
        ax.set_ylabel('$w_2$')
        plt.legend()
        fig.colorbar(CS)
        plt.tight_layout()
        if save_fig:
            fig.savefig("SVM_contour_plot")
        plt.show()
    def plot_svm_decison_boundary(self,X,w_optimized,b=0,save_fig=True):
        """
        Plot the decision boundary and margin for a classification model.
            
        Parameters:
        X (ndarray): Input feature matrix of shape (n_samples, 2), where the first column represents feature 1 and the second represents feature 2.
        w_optimized (ndarray): Optimized weight vector of shape (1, 2).
        b (float): Bias term.Defaults to 0.
        save_fig(bool): Saves the the boundary plot in the current folder.
        
        Returns:
        Displays and saves the decision boundary
        """
        fig,axis=plt.subplots()
        axis.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap="bwr", edgecolor="k")
        x0 = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200)
        x1 = -w_optimized[0][0] / w_optimized[0][1] * x0 - b / w_optimized[0][1]
        margin = 1 / np.linalg.norm(w_optimized)
        upper_margin = x1 + margin # Margin for svm is 1/|w|
        lower_margin = x1 - margin
        axis.plot(x0,x1,"-g",label="decision boundary")
        axis.plot(x0,x1+margin,"--k",label="margin")
        axis.plot(x0,x1-margin,"--k")
        fig.suptitle(" Decision Boundary plot")
        axis.set_xlabel("Feature 1")
        axis.set_ylabel("Feature 2")
        plt.tight_layout()
        axis.legend(loc="upper right")
        if save_fig:
            fig.savefig("Decision_boundary_plot")
        plt.show()

    
if __name__=="__main__":
    X,y,probabilities=preparing_data(w_true=np.array([1, 2]),b=0,n_samples=25, feature_range=(-3, 3),random_seed=42)
    svm=SVM(reg_param=0.001)

    # Optimize weights
    w_optimized = svm.optimize_weights(X, y, b=0)
    print("Optimized weights:", w_optimized)
    b_optimized=0

    # Contour plot
    svm.contour_plot_visualization(X, y, w_true=np.array([1, 2]), step=0.03)

    # Plot the decision boundary for svm
    svm.plot_svm_decison_boundary(X,w_optimized,b=0)
    
    #Determing the plot for new data
    X_test=np.array([[1,5],[2,4]])
    y_test=svm.predict(X_test)
    print(f"Predictions for the given X_test:{X_test} is: {y_test}")





