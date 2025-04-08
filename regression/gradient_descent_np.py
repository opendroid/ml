import numpy as np
import time


class GradientDescentNumpy:
    """Creates a class for gradient descent. It uses numpy for matrix
    operations.
    """

    def __init__(self, learning_rate=0.01,
                 max_iter=1000,
                 tolerance=1e-8,
                 max_tolerance=1_000):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.max_tolerance = max_tolerance
        self.dtype = np.float32
        self.params = None
        return

    def descent(self, X, y):
        """Calculate weitghts for linear regression model.

        Input:
            X <= m x n matrix (m: features, n:training samples)
            y <= m x 1 vector (targets)

        Initialize:
            W <= n x 1 params (weights), vector of zeros (or small random
                       numbers)

        Preprocess:
            Add a column of 1s to X for the intercept term

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
        y = y.astype(dtype)
        # Add a column of 1s to X for the intercept term
        X_b = np.c_[np.ones((m, 1), dtype=dtype), X]
        self.params = np.zeros((n + 1, 1), dtype=dtype)
        iter_count, tolerance = 0, dtype('inf')
        for _ in range(self.max_iter):
            # for performance use this instead of (Y_predictions - y)
            Y_predictions = X_b @ self.params
            errors = y - Y_predictions
            # Correlation between features and errors
            gradient = (1/m) * X_b.T @ errors
            # Note that actual delta is (sign is opposite)
            # (Y_predictions - y) * X_b * learning_rate
            delta = self.learning_rate * gradient
            # Note that np.linalg.norm(delta) is same as np.linalg.norm(-delta)
            tolerance = np.linalg.norm(delta)
            if tolerance < self.tolerance:
                break
            elif tolerance > self.max_tolerance:
                print(f"No convergence: {tolerance} {delta.reshape(1, -1)}")
                break
            # Update parameters by subtracting delta,
            # which is equivalent to adding -delta (see note above)
            self.params += delta
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
