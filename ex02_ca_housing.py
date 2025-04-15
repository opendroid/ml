"""scikit CA housing data model
"""
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


def main():
    ca_housing = fetch_california_housing()
    print(f"CA housing dataset: {ca_housing['DESCR']}")
    X_train, X_test, y_train, y_test = train_test_split(
        ca_housing.data, ca_housing.target, train_size=0.80, random_state=82)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_predictions = lr.predict(X_test)

    rms = mean_squared_error(y_test, y_predictions)
    print(f"Mean squared error: = {rms:0.2f}")

    accuracy = r2_score(y_test, y_predictions)
    print(f"Coefficient of determination = {accuracy:0.2f}")

    print(f"Features: {ca_housing.feature_names}")
    print(f"Target: {ca_housing.target_names}")

    # Features: ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population',
    # 'AveOccup', 'Latitude', 'Longitude']
    gilroy = [[45, 1, 7, 4, 50, 50, 36.99, -121.59]]
    result = lr.predict(gilroy)
    print(f"Gilroy house price: {result[0]:0.2f}")


if __name__ == "__main__":
    main()
