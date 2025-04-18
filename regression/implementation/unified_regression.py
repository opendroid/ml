import numpy as np
import time


class BaseRegression:
    """Creates a base class for gradient descent. It uses numpy for matrix
    operations.

    Note that it is a single feature regression model.
    """

    def __init__(self, alpha=0.01,
                 max_iter=1000,
                 tol=1e-8,
                 max_tol=1_000,
                 activation=lambda x: x):
        self.alpha = alpha  # Learning rate
        self.max_iter = max_iter  # Maximum number of iterations
        self.tol = tol  # Tolerance
        self.max_tol = max_tol
        self.coef_ = None
        self.intercept_ = None
        self.activation = activation  # Activation function
        self.training_time = 0
        self.prediction_time = 0
        self.training_iter = 0
        self.training_tolerance = 0
        self.training_loss = 0

    def fit(self, X, y):
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
        y = np.asarray(y).ravel()  # Flattens y to (m,)
        if y.shape[0] != m:
            raise ValueError(f"y shape {y.shape} does not match X rows {m}")

        # Add a column of 1s to X for the intercept term, Shape: (m, n + 1)
        X_b = np.c_[np.ones(m), X]
        self.coef_ = np.zeros(n + 1)  # Shape: (n + 1,)
        iter_count, tolerance = 0, np.inf
        for _ in range(self.max_iter):
            # for performance use this instead of (Y_predictions - y)
            # Predictions: X_b (m, n + 1) @ coef_ (n + 1,) -> (m,)
            Y_predictions = X_b @ self.coef_
            Y_predictions = self.activation(Y_predictions)
            # Errors: both predictions and y are (m,), so result is (m,)
            errors = Y_predictions - y
            # Gradient: X_b.T (n + 1, m) @ errors (m,) -> (n + 1,)
            gradient = (1/m) * (X_b.T @ errors)
            # Note that actual delta is (sign is opposite)
            # (Y_predictions - y) * X_b * learning_rate
            delta = self.alpha * gradient  # Delta: scale gradient
            # Note that np.linalg.norm(delta) is same as np.linalg.norm(-delta)
            self.training_loss = np.linalg.norm(errors)
            tolerance = np.linalg.norm(delta)
            if tolerance < self.tol:
                break
            elif tolerance > self.max_tol:
                print(f"No convergence: {tolerance} {delta.reshape(1, -1)}")
                break
            # Update parameters by subtracting delta,
            # which is equivalent to adding -delta (see note above)
            self.coef_ -= delta
            iter_count += 1

        self.training_time = time.perf_counter() - start_time
        self.training_iter = iter_count
        self.training_tolerance = tolerance
        self.intercept_ = self.coef_[0]
        return self

    def predict(self, X):
        """Predict using the linear regression model.

        Input:
            X <= m x n matrix (m: features, n:test samples)

        Output:
            predictions <= m x 1 vector (predicted targets)
            time_taken (float)
        """
        time_start = time.perf_counter()
        m = X.shape[0]
        X_b = np.c_[np.ones((m, 1)), X]
        predictions = self.activation(X_b @ self.coef_)
        predictions = predictions.reshape(-1, 1)
        self.prediction_time = time.perf_counter() - time_start
        return predictions


class LogisticRegression(BaseRegression):
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __init__(self, alpha=0.001, max_iter=1000,
                 tol=1e-8, max_tol=1_000_000):
        super().__init__(alpha, max_iter, tol, max_tol, self._sigmoid)

    def predict(self, X):
        predictions = super().predict(X)
        return (predictions > 0.5).astype(int)


class LinearRegression(BaseRegression):
    def _identity(self, x):
        return x

    def __init__(self, alpha=0.001, max_iter=1000,
                 tol=1e-8, max_tol=1_000_000):
        super().__init__(alpha, max_iter, tol,
                         max_tol,
                         activation=self._identity)
