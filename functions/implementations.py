import numpy as np
from functions.my_functions import *


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Gradient Descent for Mean Squared Error (MSE).

    Args:
        y: shape=(N, ). The target values.
        tx: shape=(N, D). The input data.
        initial_w: shape=(D, ). The initial guess for the model parameters.
        max_iters: a scalar denoting the total number of iterations of GD.
        gamma: a scalar denoting the stepsize.

    Returns:
        A tuple (w, loss) where:
            - w: the final model parameters after gradient descent (shape=(D,)).
            - loss: the final loss value after the last iteration (scalar).
    """
    # Initialize parameters
    w = initial_w

    for n_iter in range(max_iters):

        # Compute loss and gradient
        grad, err = compute_gradient(y, tx, w, loss_type="mse")

        # Update weights
        w = w - gamma * grad
        
        # Calculate loss (MSE)
        loss = calculate_mse(err)

        print(
            "GD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter + 1, ti=max_iters, l=loss, w0=w[0], w1=w[1]
            )
        )

    return w, loss  # Return the final weights and loss

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD) for linear model.

    Args:
        y: shape=(N, )
        tx: shape=(N, D). Input data.
        initial_w: shape=(D, ). The initial guess for the model parameters.
        max_iters: a scalar denoting the total number of iterations of SGD.
        gamma: a scalar denoting the stepsize.

    Returns:
        A tuple (w, loss) where:
            - w: the final model parameters after stochastic gradient descent (shape=(D,)).
            - loss: the final loss value after the last iteration (scalar).
    """
    # Initialize parameters
    w = initial_w

    # Force standard mini-batch-size 1 (sample just one datapoint).
    batch_size = 1

    # Set loss type to mse
    loss_type = "mse"

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

def least_squares(y, tx):
    """Calculate the least squares solution.
       Returns mse and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape (D,), D is the number of features.
        mse: scalar, mse loss
    """
    #optimal weights w using the normal equation: w = (XtX)^(-1)Xty <--> (XtX)w = Xty
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b) # Solve the linear equation Ax = b
    mse = compute_loss(y, tx, w) #mse
    return w, mse

def ridge_regression(y, tx, lambda_):
    """Implement ridge regression using the normal equation.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N, D), D is the number of features.
        lambda_: scalar regularization parameter.

    Returns:
        w: optimal weights, numpy array of shape (D,), D is the number of features.
        loss: Mean Squared Error, scalar.
    """
    # Number of samples and features
    N, D = tx.shape
    
    # Identity matrix for regularization term
    aI = lambda_ * np.identity(D)
    
    # Normal equation components
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    
    # Solve for weights
    w = np.linalg.solve(a, b)
    
    # Compute the loss (mse)
    loss = compute_loss(y, tx, w)

    return w, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Perform logistic regression using gradient descent.

    Args:
        y (ndarray): Labels vector of shape (N, 1), where N is the number of samples.
                     Each entry in `y` should be either 0 or 1, representing class labels.
        tx (ndarray): Feature matrix of shape (N, D), where N is the number of samples
                      and D is the number of features. Each row corresponds to the
                      features of one sample.
        initial_w (ndarray): Initial weights of shape (D, 1), where D is the number of features.
        max_iters (int): The number of iterations for the gradient descent algorithm.
        gamma (float): The learning rate, which controls the step size in gradient descent.

    Returns:
        tuple: (w, loss), where:
            - w (ndarray): The final weight vector after training, shape (D, 1).
            - loss (float): The final value of the binary cross-entropy loss.
    """
    # Initialize weights
    w = initial_w

    for n_iter in range(max_iters):
        # Compute the gradient and the loss
        grad = calculate_gradient_logistic_regression(y, tx, w)
        loss = calculate_loss_logistic_regression(y, tx, w)
        
        # Update the weights using the gradient
        w = w - gamma * grad
        
        # Print progress (optional)
        print(f"Iteration {n_iter+1}/{max_iters}, Loss: {loss}, Weights: {w.flatten()}")
    
    # Return the final weights and the final loss value
    return w, loss

def reg_logistic_regression(y, tx, initial_w, max_iters, gamma, lambda_):
    """
    Perform logistic regression using gradient descent with L2 regularization.

    Args:
        y (ndarray): Labels vector of shape (N, 1).
        tx (ndarray): Feature matrix of shape (N, D).
        initial_w (ndarray): Initial weights of shape (D, 1).
        max_iters (int): The number of iterations for gradient descent.
        gamma (float): The learning rate.
        lambda_ (float): The regularization parameter for L2 regularization.

    Returns:
        tuple: (w, loss), where:
            - w (ndarray): The final weight vector after training, shape (D, 1).
            - loss (float): The final value of the penalized binary cross-entropy loss.
    """
    # Initialize weights
    w = initial_w

    for n_iter in range(max_iters):
        # Compute the loss and gradient
        loss, gradient = calculate_loss_grad_penalized_logistic_regression(y, tx, w, lambda_)

        # Update the weights using the gradient
        w -= gamma * gradient
        
        # Print progress (optional)
        print(f"Iteration {n_iter + 1}/{max_iters}, Loss: {loss:.4f}, Weights: {w.flatten()}")

    return w, loss





