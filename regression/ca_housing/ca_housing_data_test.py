import pytest
from ca_housing_data import CAHousingData


def test_data_shape():
    data = CAHousingData()
    assert data.data.shape == (20640, 10)


def test_numerical_features():
    data = CAHousingData()
    numerical_features = data.numerical_features()
    expected_features = ['longitude',
                         'latitude',
                         'housing_median_age',
                         'total_rooms',
                         'total_bedrooms',
                         'population',
                         'households',
                         'median_income']
    assert set(numerical_features) == set(expected_features)


def test_categorical_features():
    data = CAHousingData()
    assert data.categorical_features() == ['ocean_proximity']


def test_ocean_categories():
    data = CAHousingData()

    # Check if ocean_categories is a list
    assert isinstance(data.ocean_categories(), list)

    # Check if ocean_categories is not empty
    assert len(data.ocean_categories()) > 0

    # Check if ocean_categories is a list of strings
    assert all(isinstance(category, str)
               for category in data.ocean_categories())

    categories = ['NEAR BAY', '<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'ISLAND']
    assert set(data.ocean_categories()) == set(categories)


def test_features():
    data = CAHousingData()
    assert data.features().shape == (20640, 9)


def test_labels():
    data = CAHousingData()
    assert data.labels().shape == (20640,)


if __name__ == "__main__":
    pytest.main()
