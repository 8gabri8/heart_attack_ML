import numpy as np
from my_functions import calculate_mae, compute_gradient

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
