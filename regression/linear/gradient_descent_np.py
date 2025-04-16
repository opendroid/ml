import numpy as np
import time


class GradientDescentBase:
    """Creates a base class for gradient descent. It uses numpy for matrix
    operations.
    """

    def __init__(self, alpha=0.01,
                 max_iter=1000,
                 tol=1e-8,
                 max_tol=1_000,
                 activation=None):
        self.alpha = alpha  # Learning rate
        self.max_iter = max_iter  # Maximum number of iterations
        self.tol = tol  # Tolerance
        self.max_tol = max_tol
        self.dtype = np.float32
        self.params = None
        self.activation = activation  # Activation function

    def descent(self, X, y):
        """Calculate weitghts for linear regression model.

        Input:
            X <= m x n matrix (m: training samples, n:features)
            y <= m x 1 vector (targets)

        Initialize:
            W <= n x 1 params (weights), vector of zeros (or small random
                       numbers)

        Preprocess:
            Add a column of 1s to X for the intercept term
            Add a column of 1s to W as weight of intercept term

        Loop (for num_iters):
            predictions <= X ⋅ θ
            errors <= predictions - y
            gradients <= (1/m) ⋅ Xt⋅ errors
            W <= W - alpha ⋅ gradients

        Output:
            W (final params including intercept)
            time_taken (float)
        """
        start_time = time.perf_counter()
        m, n = X.shape
        dtype = self.dtype
        X = X.astype(dtype)
        y = np.asarray(y, dtype=self.dtype).ravel()  # Flattens y to (m,)
        if y.shape[0] != m:
            raise ValueError(f"y shape {y.shape} does not match X rows {m}")
        print(f"X: {X.shape}, y: {y.shape}")
        # Add a column of 1s to X for the intercept term, Shape: (m, n + 1)
        X_b = np.c_[np.ones(m, dtype=dtype), X]
        self.params = np.zeros(n + 1, dtype=dtype)  # Shape: (n + 1,)
        iter_count, tolerance = 0, dtype('inf')
        for _ in range(self.max_iter):
            # for performance use this instead of (Y_predictions - y)
            # Predictions: X_b (m, n + 1) @ params (n + 1,) -> (m,)
            Y_predictions = X_b @ self.params
            # Errors: both predictions and y are (m,), so result is (m,)
            errors = Y_predictions - y
            # Correlation between features and errors
            # Gradient: X_b.T (n + 1, m) @ errors (m,) -> (n + 1,)
            gradient = (1/m) * X_b.T @ errors
            # Note that actual delta is (sign is opposite)
            # (Y_predictions - y) * X_b * learning_rate
            delta = self.learning_rate * gradient  # Delta: scale gradient
            # Note that np.linalg.norm(delta) is same as np.linalg.norm(-delta)
            tolerance = np.linalg.norm(delta)
            if tolerance < self.tolerance:
                break
            elif tolerance > self.max_tolerance:
                print(f"No convergence: {tolerance} {delta.reshape(1, -1)}")
                break
            # Update parameters by subtracting delta,
            # which is equivalent to adding -delta (see note above)
            self.params -= delta
            iter_count += 1
        time_taken = time.perf_counter() - start_time

        return time_taken, iter_count, tolerance

    def predict(self, X):
        """Predict using the linear regression model.

        Input:
            X <= m x n matrix (m: features, n:training samples)

        Output:
            predictions <= m x 1 vector (predicted targets)
            time_taken (float)
        """
        time_start = time.perf_counter()
        dtype = self.dtype
        m = X.shape[0]
        X = X.astype(dtype)
        X_b = np.c_[np.ones((m, 1), dtype=dtype), X]
        predictions = X_b @ self.params
        time_taken = time.perf_counter() - time_start
        return predictions, time_taken
