import time
import torch
import torch.profiler


class GradientDescentTorch:
    """Creates a class for gradient descent. It uses torch for matrix
    operations.
    """

    def __init__(self, learning_rate=0.01, max_iter=1000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.dtype = torch.float32
        self.params = None
        # set device to GPU if available, otherwise CPU
        self.device = torch.device("mps"
                                   if torch.backends.mps.is_available()
                                   else "cpu")

    def descent(self, X, y):
        """Calculate weights for linear regression mode
        Input:
            X <= m x n matrix (m: features, n:training samples)
            y <= m x 1 vector (targets)

        Initialize:
            P <= n x 1 params (weights), vector of zeros

        Preprocess:
            Add a column of 1s to X for the intercept term

        Loop (for num_iters):
            predictions <= X ⋅ θ
            errors <= predictions - y
            gradients <= -(1/m) ⋅ Xt⋅ errors
            P <= P - alpha ⋅ gradients

        Output:
            P (final params including intercept)
            time_taken (float)
        """
        start_time = time.perf_counter()
        device = self.device
        dtype = self.dtype
        m, n = X.shape
        X = X.to(device, dtype=dtype)
        y = y.to(device, dtype=dtype)

        # Add a column of 1s to X for the intercept term
        ones = torch.ones((m, 1), dtype=dtype, device=device)
        X_b = torch.cat((ones, X), dim=1)  # on device
        self.params = torch.zeros((n + 1, 1), dtype=dtype, device=device)
        tolerance = torch.tensor(float('inf'), dtype=dtype, device=device)
        iter_count = 0

        for _ in range(self.max_iter):
            Y_predictions = X_b @ self.params
            # for performance use this instead of (Y_predictions - y)
            errors = y - Y_predictions
            # Correlation between features and errors
            gradient = (1.0/m) * X_b.T @ errors
            # Note that actual delta is (sign is opposite)
            # (Y_predictions - y) * X_b * learning_rate
            delta = self.learning_rate * gradient
            # Note that torch.norm(delta) is same as torch.norm(-delta)
            tolerance = torch.norm(delta)
            if tolerance < self.tolerance:
                break
            elif tolerance > 1000:
                tolerance = torch.tensor(
                    float('inf'), dtype=dtype, device=device)
                print(f"Will not converge: {delta}")
                break
            # Update parameters by subtracting delta,
            # which is equivalent to adding -delta (see note above)
            self.params += delta
            iter_count += 1

        time_taken = time.perf_counter() - start_time
        # Move self.params back to CPU
        self.params = self.params.cpu()

        return time_taken, iter_count, tolerance.cpu()

    def predict(self, X):
        """Predict using the linear regression model.

        Input:
            X <= m x n matrix (m: features, n:training samples)

        Output:
            predictions <= m x 1 vector (predicted targets)
            time_taken (float)
        """
        time_start = time.perf_counter()
        device = self.device
        dtype = self.dtype  # Same everywhere
        m = X.shape[0]
        X = X.to(device, dtype=dtype)
        ones = torch.ones((m, 1), dtype=dtype, device=device)
        X_b = torch.cat((ones, X), dim=1)  # on device
        self.params = self.params.to(device)
        predictions = X_b @ self.params
        time_taken = time.perf_counter() - time_start
        self.params = self.params.cpu()
        return predictions, time_taken
