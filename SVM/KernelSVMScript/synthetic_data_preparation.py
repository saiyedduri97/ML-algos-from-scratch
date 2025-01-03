import numpy as np



def preparing_data(w_true, b=0,n_samples=25, feature_range=(-3, 3),random_seed=42):
    """
    Generate synthetic data for binary classification with labels in {-1, +1}.
    
    Parameters: 
        w_true (numpy array): True weight vector used for computing probabilities.
        b (float): Bias term for the decision function.
        n_samples (int): Number of samples to generate.
        feature_range (tuple): Range for the feature values.
        random_seed (int): Seed for random number generation.
    Returns:
        X (numpy array): Feature matrix of shape (n_samples, 2).
        y (numpy array): Binary labels of shape (n_samples, 1), in the range {-1, +1}.
        probabilities (numpy array): Probabilities of the positive class for each sample.
    
    Example:
        X, y, probabilities = preparing_data(seed=42, n_samples=25, feature_range=(-3, 3), w_true=np.array([1, 2]), b=0)
    """
    rng = np.random.default_rng(seed=random_seed)
    
    # Generate random feature samples
    X = rng.uniform(low=feature_range[0], high=feature_range[1], size=(n_samples, 2))
    
    # Compute probabilities using the sigmoid function
    z = (X @ w_true.T + b).reshape(-1, 1)
    sigmoid = lambda z: 1 / (1 + np.exp(-z))
    probabilities = sigmoid(z)
    
    # Generate binary labels based on probabilities
    y = 2 * (rng.uniform(0, 1, size=(n_samples, 1)) <= probabilities) - 1
    
    return X, y, probabilities
