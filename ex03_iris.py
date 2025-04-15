"""Example SVM and LR classifiers.
Book:
"""

from sklearn.datasets import load_iris, get_data_home
from sklearn.model_selection import train_test_split
from sklearn import svm, linear_model
from sklearn.metrics import accuracy_score


def main():
    # Download the iris flower data set. By default the data is cached
    # in ~/scikit_learn_data
    iris_data = load_iris()

    # Print dataset properties
    print(f"{iris_data['DESCR']}")
    print(f"Classifications: {iris_data['target_names']}")
    print(f"data home: {get_data_home()}")

    # Create 95% trainig set and 5% test set
    X_train, X_test, y_train, y_test = train_test_split(iris_data['data'],
                                                        iris_data['target'],
                                                        train_size=0.8,
                                                        random_state=82)
    # Print samples
    print(f"X_train: {len(X_train)}\n: {X_train[:5]}")
    print(f"X_test: {len(X_test)}\n:{X_test[:5]}")
    print(f"y_train: {len(y_train)}:{y_train[:5]}")
    print(f"y_test: {len(y_test)}:{y_test[:5]}")

    # Train SVM
    svm_classifier = svm.SVC(kernel="linear")
    svm_classifier.fit(X_train, y_train)

    # Train LR
    lr_classifier = linear_model.LogisticRegression()
    lr_classifier.fit(X_train, y_train)

    # Train a LinearRegression  model
    linear_regression_classifier = linear_model.LinearRegression()
    linear_regression_classifier.fit(X_train, y_train)

    # Test SVM, Logistic and Linear regressions.
    svm_predictions = svm_classifier.predict(X_test)
    svm_accuracy = accuracy_score(y_test, svm_predictions)
    lr1_predictions = lr_classifier.predict(X_test)
    lr1_accuracy = accuracy_score(y_test, lr1_predictions)
    lr2_predictions_ = linear_regression_classifier.predict(X_test)

    def convert(x): return (0 if x < 0.5 else 1 if x < 1.5 else 2)

    # LR is continous value, convert to index
    lr2_predictions = [convert(x) for x in lr2_predictions_]

    # Print test results
    for i in range(len(y_test)):
        expected = iris_data['target_names'][y_test[i]]
        s_p = svm_predictions[i]
        s_n = iris_data['target_names'][s_p]
        lr1_p = lr1_predictions[i]
        lr1_n = iris_data['target_names'][lr1_p]
        lr2_p = lr2_predictions[i]
        lr2_n = iris_data['target_names'][lr2_p]
        print(f"{X_test[i]}:{expected:^12}:{s_n:^12}:{lr1_n:^12}:{lr2_n:^12}")

    print(f"SVM accuracy: {svm_accuracy:0.2f}")
    print(f"LR Accuracy: {lr1_accuracy:0.2f}")


if __name__ == "__main__":
    main()
