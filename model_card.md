# Model Card

## Model Details

- **Model Name**: Random Forest Classifier



## Intended Use

**Primary Use Case**:

The primary use case for this model is to assist in predicting individuals' income levels based on census features.

**Potential Users**:

- Data Analysts
- Data Scientists
- Charitable Organizations
- Marketing Teams
- Researchers

## Training Data

**Data Source**:

The model was trained on a dataset obtained from the UCI Machine Learning Repository. The dataset contains various demographic features such as age, education, marital status, etc., as well as the target variable indicating salary class (<=50k or >50k).

**Data Preprocessing**:

The training data underwent preprocessing steps that included handling missing values, encoding categorical features, and splitting the data into training and validation sets.
## Evaluation Data

The evaluation data is a subset of the same dataset used for training. Here, I split the data to training set and evaluation set with the proportion as 80/20. It was reserved for model evaluation to assess its generalization performance.


## Metrics
**Evaluation Metrics**:

- Precision: The ratio of true positive predictions to the total predicted positives.
- Recall: The ratio of true positive predictions to the total actual positives.
- F-beta: The weight harmonic of precision and recall.

**Model Performance**:

- Precision: 0.74
- Recall: 0.66
- F1-beta: 0.7

