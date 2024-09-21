import numpy as np

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Create a minibatch iterator for a dataset.

    Args:
        y: shape=(N,). Target values.
        tx: shape=(N, D). Input data.
        batch_size: Number of samples per batch.
        num_batches: Number of batches to generate.
        shuffle: If True, shuffle data before splitting into batches.

    Returns:
        A generator yielding tuples (minibatch_y, minibatch_tx), where each batch contains 
        `batch_size` samples, except possibly the last batch.
    """

    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx

    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1 / 2 * np.mean(e**2) # 0.5 factor as in Lessons

def calculate_mae(e):
    """Calculate the mae for vector e."""
    return np.mean(np.abs(e))

def compute_loss(y, tx, w, loss_type="mse"):
    """Calculate the loss of a linear model using either MSE or MAE.

    Args:
        y: shape=(N, )
        tx: shape=(N, D)
        w: shape=(D,). The vector of model parameters.
        loss_type: string, either 'mse' or 'mae', specifies which loss function to use.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    e = y - tx.dot(w) # label - prediction of the linear model
    if loss_type == "mse":
        return calculate_mse(e)
    elif loss_type == "mae":
        return calculate_mae(e)
    else:
        raise ValueError("Invalid loss_type. Choose 'mse' or 'mae'.")

def compute_gradient(y, tx, w, loss_type="mse"):
    """Computes the gradient at w for either MSE or MAE loss.

    Args:
        y: shape=(N, )
        tx: shape=(N, D). Input data.
        w: shape=(D, ). The vector of model parameters.
        loss_type: string, either 'mse' or 'mae', specifies which loss function to use.

    Returns:
        A tuple containing:
            - grad: An array of shape (D, ) (same shape as w), containing the gradient of the loss at w.
            - err: The error vector (shape=(N,)).
    """
    err = y - tx.dot(w)
    
    if loss_type == "mse":
        grad = -tx.T.dot(err) / len(err) # -(1/N)X^t(e)
    elif loss_type == "mae":
        grad = -tx.T.dot(np.sign(err)) / len(err) # -(1/N)X^t sign(e)
    else:
        raise ValueError("Invalid loss_type. Choose 'mse' or 'mae'.")
    
    return grad, err  # Return as a tuple

def gradient_descent(y, tx, initial_w, max_iters, gamma, loss_type="mse"):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: shape=(N, )
        tx: shape=(N, D). Input data.
        initial_w: shape=(D, ). The initial guess for the model parameters.
        max_iters: a scalar denoting the total number of iterations of GD.
        gamma: a scalar denoting the stepsize.
        loss_type: string, either 'mse' or 'mae', specifies which loss function to use.

    Returns:
        A tuple (w, loss) where:
            - w: the final model parameters after gradient descent (shape=(D,)).
            - loss: the final loss value after the last iteration (scalar).
    """
    # Initialize parameters
    w = initial_w

    for n_iter in range(max_iters):
        # Compute loss and gradient
        grad, err = compute_gradient(y, tx, w, loss_type)
        
        # Compute loss based on the selected loss type
        if loss_type == "mse":
            loss = calculate_mse(err)
        elif loss_type == "mae":
            loss = calculate_mae(err)
        else:
            raise ValueError("Invalid loss_type. Choose 'mse' or 'mae'.")

        # Update w by gradient descent
        w = w - gamma * grad

        print(
            "GD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter + 1, ti=max_iters, l=loss, w0=w[0], w1=w[1]
            )
        )

    return w, loss  # Return the final weights and loss

def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma, loss_type="mse"):
    """The Stochastic Gradient Descent algorithm (SGD) for linear model.

    Args:
        y: shape=(N, )
        tx: shape=(N, D). Input data.
        initial_w: shape=(D, ). The initial guess for the model parameters.
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient.
        max_iters: a scalar denoting the total number of iterations of SGD.
        gamma: a scalar denoting the stepsize.
        loss_type: string, either 'mse' or 'mae', specifies which loss function to use.

    Returns:
        A tuple (w, loss) where:
            - w: the final model parameters after stochastic gradient descent (shape=(D,)).
            - loss: the final loss value after the last iteration (scalar).
    """
    # Initialize parameters
    w = initial_w

    for n_iter in range(max_iters):
        
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            # Compute stochastic gradient and loss
            grad, _ = compute_gradient(y_batch, tx_batch, w, loss_type)
            # Update w through the stochastic gradient update
            w = w - gamma * grad
            
            # Calculate loss for the current weights
            loss = compute_loss(y_batch, tx_batch, w, loss_type)

        print(
            "SGD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter + 1, ti=max_iters, l=loss, w0=w[0], w1=w[1]
            )
        )

    return w, loss  # Return the final weights and loss


def pca(x_train, variance_threshold):
    """Perform PCA on the given dataset and return components that explain
    at least the specified percentage of variance.

    Args:
        x_train (np.ndarray): Input data, shape (N, D) where N is the number of samples and D is the number of features.
        variance_threshold (float): The desired percentage of variance to be explained (0 to 1).

    Returns:
        np.ndarray: The transformed data, shape (N, k) where k is the number of components selected.
        np.ndarray: The eigenvalues, shape (D,).
        np.ndarray: The eigenvectors, shape (D, D).
    """
    # Step 1: Center the data
    x_centered = x_train - np.mean(x_train, axis=0)

    # Step 2: Compute the covariance matrix
    covariance_matrix = np.cov(x_centered, rowvar=False)

    # Step 3: Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Step 4: Sort the eigenvalues and eigenvectors
    sorted_indices = np.argsort(eigenvalues)[::-1]  # Indices of eigenvalues in descending order
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Step 5: Calculate the explained variance
    explained_variance = sorted_eigenvalues / np.sum(sorted_eigenvalues)
    
    # Step 6: Cumulative explained variance
    cumulative_variance = np.cumsum(explained_variance)

    # Step 7: Determine the number of components to retain
    num_components = np.argmax(cumulative_variance >= variance_threshold) + 1

    # Step 8: Project the data onto the selected principal components
    selected_eigenvectors = sorted_eigenvectors[:, :num_components]
    x_transformed = np.dot(x_centered, selected_eigenvectors)

    return x_transformed

def build_poly(x, degree):
    """Generate polynomial basis functions for input data x, for j=0 up to j=degree.

    Args:
        x (np.ndarray): Input data of shape (N, D) where N is the number of samples and D is the number of features.
        degree (int): The maximum degree of polynomial features to generate.

    Returns:
        np.ndarray: Polynomial feature matrix of shape (N, D * degree + 1).
    """
    N, D = x.shape  # Number of samples and features
    poly = np.ones((N, 1))  # Start with the bias term (degree 0)

    for deg in range(1, degree + 1):
        for feature_index in range(D):
            # Compute polynomial features for each feature
            poly_feature = np.power(x[:, feature_index], deg)
            poly = np.c_[poly, poly_feature]

    return poly
