import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from ReliefF import ReliefF
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

plt.style.use('ggplot')


def read_diabetes_data():

    diabetes_df = pd.read_csv("dataset_diabetes/diabetic_data.csv")

    return diabetes_df


def data_cleaning(diabetes_df):
    remove_columns = ['encounter_id', 'patient_nbr', 'admission_source_id', 'payer_code', 'readmitted']

    features_df = diabetes_df.drop(remove_columns, axis=1)
    return features_df



def label_encoding(features_df):

    le_features = ['race', 'gender', 'age', 'weight', 'medical_specialty',
       'diag_1',
       'diag_2', 'diag_3', 'max_glu_serum', 'A1Cresult',
       'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
       'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',
       'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
       'tolazamide', 'examide', 'citoglipton', 'insulin',
       'glyburide-metformin', 'glipizide-metformin',
       'glimepiride-pioglitazone', 'metformin-rosiglitazone',
       'metformin-pioglitazone', 'change', 'diabetesMed']


    le = preprocessing.LabelEncoder()

    features_df[le_features] = features_df[le_features].apply(le.fit_transform)

    return features_df


def create_target(diabetes_df):

    y = (diabetes_df['readmitted'] != 'NO').astype(int).values

    return y


def relief_algorithm(features_df, y):

    data = features_df.values
    
    fs = ReliefF(n_neighbors=5, n_features_to_keep=36)
    X_relieff = fs.fit_transform(data, y)
    top_features = features_df.columns[fs.top_features]

    return top_features


def  model_training_random_forest(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=0,  n_jobs=-1)
    #model.fit(X_train, y_train)

    scoring = ['accuracy', 'recall', 'f1']
    cv_scores = cross_validate(model, X_train, y_train, cv=5,
                            scoring=scoring, return_train_score=True, n_jobs=-1)

    return cv_scores


def model_training_svm(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.3, random_state=42)
    model = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    #model.fit(X_train, y_train)

    scoring = ['accuracy', 'recall', 'f1']
    cv_scores = cross_validate(model, X_train, y_train, cv=5,
                            scoring=scoring, return_train_score=True, n_jobs=-1)

    return cv_scores


def plot_cv_scores(cv_scores):

    plt.plot(cv_scores['test_accuracy'], 'o-')
    plt.plot(cv_scores['test_recall'], 'o-')
    plt.plot(cv_scores['test_f1'], 'o-')
    plt.xlabel('K-fold CV Set')
    plt.legend(['Accuracy', 'Recall', 'F1 Score'])

    return


