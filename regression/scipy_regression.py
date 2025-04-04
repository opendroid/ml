# Create a simple linear regression model

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def linear_regression(x, y):
    # Use scipy.stats.linregress to fit a linear regression model
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    # Print the results
    print(f"Slope: {slope}")
    print(f"Intercept: {intercept}")
    print(f"R-value: {r_value}")
    print(f"P-value: {p_value}")
    print(f"Standard Error: {std_err}")
    return slope, intercept


def plot_regression(x, y, slope, intercept):
    # Create a scatter plot of the data
    plt.scatter(x, y, label='Data')

    # Plot the regression line
    plt.plot(x, slope * x + intercept, 'r', label='Regression Line')

    # Add labels and legend
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Generate some sample data
    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    y = 2 * x + np.random.randn(100) * 2

    # Fit the linear regression model
    slope, intercept = linear_regression(x, y)

    # Plot the regression line
    plot_regression(x, y, slope, intercept)
