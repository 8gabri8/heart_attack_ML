import numpy as np
from my_functions import calculate_mae, compute_gradient
from my_functions import calculate_mae, compute_gradient, batch_iter, compute_loss


def mean_absolute_error_gd(y, tx, initial_w, max_iters, gamma):
    """Gradient Descent for Mean Absolute Error (MAE).

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
        grad, err = compute_gradient(y, tx, w, loss_type="mae")

        # Update weights
        w = w - gamma * grad
        
        # Calculate loss (MAE)
        loss = calculate_mae(err)

        print(
            "GD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter + 1, ti=max_iters, l=loss, w0=w[0], w1=w[1]
            )
        )

    return w, loss  # Return the final weights and loss

def mean_absolute_error_sgd(y, tx, initial_w, max_iters, gamma):
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

    # Set loss type to mae
    loss_type = "mae"

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
