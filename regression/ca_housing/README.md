# California Housing Price Prediction

Practies [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://a.co/d/a1X8ZuD)
    by Aurélien Géron
This exampe analyzes the classification problem of predicting the median house value in California.
We shall look at the following topics:
- Data loading and exploration
- Data cleaning and preprocessing
- Model building and evaluation
- Model selection and tuning
- Predictions


## Data

The data is from the California Housing Prices dataset.

The dataset is available at the [handson-ml2](https://github.com/ageron/handson-ml2/blob/master/datasets/housing/housing.csv) github repository.
The[README](https://github.com/ageron/handson-ml2/tree/master/datasets/housing) describes the dataset. Few changes from the
orginal dataset are:
 - 207 samples were randomly dropped from the original dataset.
 - Categorical feature calles ocean_proximity was added to the dataset.

Each row in the dataset represents a district (called a census block group). The dataset has the following features:

 The dataset has the following features:
 - longitude
 - latitude
 - housing_median_age
 - total_rooms
 - total_bedrooms
 - population
 - households
 - median_income: in 10_000
 - median_house_value <== Target label
 - ocean_proximity (Added)

 ## Steps

Follow the steps in the notebook.
1. Step 1: Understand the data, look at the data types, look at the distribution of the data.
    - Data types, missing values, distribution of the data.
    - Boxplot, histogram, scatter plots
    - Income distribution
    - Plot data on the map
    - Finally save to paquet format
2. Step 2: Clean and preprocess the data. Try some default models. Test with KFold=10.
    - Prepare the data for model training:
        - Read paquet file, separate features and labels
        - Impute missing values with median of the column
        - Add three new features:
            - rooms_per_household
            - bedrooms_per_room
            - population_per_household
        - Standardize the features
        - Apple one-hot encoding to the ocean_proximity feature
        - Data is ready for trying some default models
    - Try some default models.
        - Models trained on all the data, and tested with test samples:
        - The models are trained with X_train and tested on stratified test samples.
        - Models tried are:
            - Linear Regression, Lasso, Ridge
            - Decision Tree, Random Forest
        - Models cross-validated with KFold=10:
            - Linear Regression, Lasso, Ridge
            - Decision Tree, Random Forest
3. Step 3: Fine-tune various models.
    - GridSearchCV
    - RandomizedSearchCV

Observations on step 2 models:
- Linear Regression, Lasso, Ridge are producing similar results.
- Decision Tree trained on all the data is overfitting. Zero error.
- Decision Tree X_train, X_test is producing worse than Linear Regression.
- Random Forest produces best results. On Cross-validation and on test data best mean and sted deviation.

 ## Acknowledgements

 This example is based on the following sources:
 - [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://a.co/d/a1X8ZuD)
    by Aurélien Géron
 - [ML2 Repo](https://github.com/ageron/handson-ml2)
 - [ML3 Latest Repo](https://github.com/ageron/handson-ml3/tree/main)
 - ChatGPT, Cursor, Grok, Gemini, Claude
