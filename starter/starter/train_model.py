import logging
# Script to train machine learning model.
import pandas as pd
from starter.ml.data import process_data
from starter.ml.model import train_model
from sklearn.model_selection import train_test_split
import joblib

from ml.model import compute_model_metrics

def slice_census(data, cat_features):
    """ Function for evaluate model on slice of dataset """
    
    train, test = train_test_split(data, test_size=0.20)

    model = joblib.load('model/model.pkl')
    encoder = joblib.load('model/encoder.pkl')
    lb = joblib.load('model/lb.pkl')
    slice_result = {'feature': [], 'category': [], 'precision': [], 'recall': [], 'Fbeta': []}

    for cat in cat_features:
        for cls in test[cat].unique():
            df_temp = test[test[cat] == cls]

            X_test, y_test, _, _ = process_data(
                df_temp, categorical_features=cat_features, label='salary', training=False,
                encoder=encoder, lb=lb
            )

            y_pred = model.predict(X_test)

            precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
            slice_result['feature'].append(cat)
            slice_result['category'].append(cls)
            slice_result['precision'].append(precision)
            slice_result['recall'].append(recall)
            slice_result['Fbeta'].append(fbeta)
    
    df = pd.DataFrame.from_dict(slice_result)
    df.to_csv('slice_output.txt', index=False)

# Initialize logging
logging.basicConfig(filename='logging.log',
                    level=logging.INFO,
                    filemode='a',
                    format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Add code to load in the data.
logger.info('Read data')

datapath = "../data/census.csv"
data = pd.read_csv(datapath)


# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data,
                            test_size=0.20,
                            stratify=data['salary']
                            )

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
X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=cat_features, 
    label="salary",
    training=True
)

# Proces the test data with the process_data function.

X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb
)

# Train and save a model.

logger.info('Training Random Forest Classifier')
model = train_model(X_train, y_train)
# save model
logger.info('Saving model')
joblib.dump(model, 'model/model.pkl')
joblib.dump(encoder, 'model/encoder.pkl')
joblib.dump(lb, 'model/lb.pkl')

