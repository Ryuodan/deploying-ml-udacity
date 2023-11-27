from main import app
from fastapi.testclient import TestClient
from starter.ml.data import process_data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import pytest
import pandas as pd
import sys
#os.pa

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


def test_data():
    """ Test data csv """

    data = pd.read_csv('data/census_clean.csv')
    assert data.shape[0] > 0


def test_process_data():
    """ Test process data """

    data = pd.read_csv('data/census_clean.csv')
    train, test = train_test_split(data, test_size=0.20)

    X_train, y_train, _, _ = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    assert X_train.shape[0] == y_train.shape[0]


def test_model():
    """ Test Random Forest model """

    model = joblib.load('model/model.pkl')
    assert isinstance(model, RandomForestClassifier)


client = TestClient(app)


def test_get():
    """ Test the root welcome page """
    r = client.get('/')
    assert r.status_code == 200
    assert r.json() == {'message': 'Hello World'}


def test_post_above():
    """ Test the output for salary is >50k """

    r = client.post('/prediction', json={
        "age": 31,
        "workclass": "Private",
        "fnlgt": 45781,
        "education": "Masters",
        "education-num": 14,
        "marital-status": "Never-married",
        "occupation": "Prof-specialty",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Female",
        "capital-gain": 14084,
        "capital-loss": 0,
        "hours-per-week": 50,
        "native-country": "United-States"
    })

    assert r.status_code == 200
    assert r.json() == {'Predicted Income': ' >50K'}


def test_post_below():
    """ Test the output for salary is <50k """

    r = client.post('/prediction', json={
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    })

    assert r.status_code == 200
    assert r.json() == {'Predicted Income': ' <=50K'}
