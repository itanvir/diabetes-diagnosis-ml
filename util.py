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
from keras.utils.np_utils import to_categorical  
from sklearn.metrics import classification_report
import tensorflow as tf 
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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=0,  n_jobs=-1)
    #model.fit(X_train, y_train)

    scoring = ['accuracy', 'recall', 'f1']
    cv_scores = cross_validate(model, X_train, y_train, cv=5,
                            scoring=scoring, return_train_score=True, n_jobs=-1)

    return cv_scores


def model_training_svm(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    #model.fit(X_train, y_train)

    scoring = ['accuracy', 'recall', 'f1']
    cv_scores = cross_validate(model, X_train, y_train, cv=5,
                            scoring=scoring, return_train_score=True, n_jobs=-1)

    return cv_scores


def plot_cv_scores(cv_scores):

    plt.figure()
    plt.plot(cv_scores['test_accuracy'], 'o-')
    plt.plot(cv_scores['test_recall'], 'o-')
    plt.plot(cv_scores['test_f1'], 'o-')
    plt.xlabel('K-fold CV Set')
    plt.legend(['Test Accuracy', 'Test Recall', 'Test F1 Score'])

    plt.figure()
    plt.plot(cv_scores['train_accuracy'], 'o-')
    plt.plot(cv_scores['train_recall'], 'o-')
    plt.plot(cv_scores['train_f1'], 'o-')
    plt.xlabel('K-fold CV Set')
    plt.legend(['TrainAccuracy', 'Train Recall', 'Train F1 Score'])

    return


def plot_history(history):

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.xlabel("Epochs")
    plt.legend(['Train Accuracy', 'Val Accuracy'])

    return None


def model_training_cnn(X, y):

    # To categorical
    y = to_categorical(y, num_classes=2)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    input_shape=(6, 6, 1)
    num_classes = 2

    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=input_shape),
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),
            #tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            #tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    print (model.summary())

    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=["accuracy"])


    callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        'models/model.hdf5',
        save_best_only=True,  # Only save a model if `val_loss` has improved.
        monitor="val_loss",
        verbose=1,
        save_format='h5',
    )
]
    history = model.fit(
        X_train, y_train, epochs=50, batch_size=64, callbacks=callbacks, validation_split=0.3
    )

    # Validation on test set
    y_pred_proba = model.predict(X_test)[:, 1]
    y_pred = (y_pred_proba>0.5).astype(int)
    y_true = y_test[:, 1]
    
    print (classification_report(y_true, y_pred))

    return history



def get_intermediate_layer_output(X):

    model = tf.keras.models.load_model('models/model.hdf5')

    aux_model = tf.keras.Model(inputs=model.inputs,
                           outputs=model.outputs + [model.layers[3].output])

    # Access both the final and intermediate output of the original model
    final_output, X_embedding = aux_model.predict(X)

    return X_embedding