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
from IPython.display import display
import seaborn as sns
import scipy as sp
from imblearn.over_sampling import SMOTE

plt.style.use('ggplot')

def generate_training_data():
    df = pd.read_csv("dataset_diabetes/diabetic_data.csv")

    # columns with PII
    df = df.drop(['encounter_id', 'patient_nbr', 'payer_code'], axis = 1)
    # columns with large number of missing values (~97%)
    df = df.drop(['weight'], axis = 1)
    # columns in which almost all records (99.95%) have the same value
    df = df.drop(['citoglipton', 'examide', 'acetohexamide', 'tolbutamide', 'miglitol',
        'troglitazone', 'tolazamide', 'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone',
        'metformin-pioglitazone'], axis = 1)

    # rows with invalid values
    drop_Idx = set(df['diag_1'][df['diag_1'] == '?'].index)
    drop_Idx = drop_Idx.union(set(df['race'][df['race'] == '?'].index))
    drop_Idx = drop_Idx.union(set(df[df['discharge_disposition_id'] == 11].index))
    drop_Idx = drop_Idx.union(set(df['gender'][df['gender'] == 'Unknown/Invalid'].index))
    new_Idx = list(set(df.index) - set(drop_Idx))
    df = df.iloc[new_Idx]
    print("After removing rows and columns: ", df.shape)

    # Subjective and non-unique feature engineering, depending on knowledge of health care services.
    # Customized based on https://www.kaggle.com/code/iabhishekofficial/prediction-on-hospital-readmission
    # number of medication changes
    medications = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'glipizide', 'glyburide',
        'pioglitazone', 'rosiglitazone', 'acarbose', 'insulin', 'glyburide-metformin']
    for col in medications:
        colname = str(col) + '_temp'
        df[colname] = df[col].apply(lambda x: 0 if (x == 'No' or x == 'Steady') else 1)
    df['num_med_change'] = 0
    for col in medications:
        colname = str(col) + '_temp'
        df['num_med_change'] = df['num_med_change'] + df[colname]
        del df[colname]

    # Re-encode admission type, discharge type and admission source into fewer categories
    admission_type_map = [(2,1), (7,1), (6,5), (8,5)]
    for p in admission_type_map:
        df['admission_type_id'] = df['admission_type_id'].replace(p[0], p[1])
    
    discharge_disposition_map = [(6,1), (8,1), (9,1), (13,1), (3,2), (4,2), (5,2), (14,2), (22,2), (23,2), 
        (24,2), (12,10), (15,10), (16,10), (17,10), (25,18), (26,18)]
    for p in discharge_disposition_map:
        df['discharge_disposition_id'] = df['discharge_disposition_id'].replace(p[0], p[1])

    admission_source_map = [(2,1), (3,1), (5,4), (6,4), (10,4), (22,4), (25,4), (15,9), (17,9), (20,9),
        (21,9), (13,11), (14,11)]
    for p in admission_source_map:
        df['admission_source_id'] = df['admission_source_id'].replace(p[0], p[1])

    # Encode string values
    df['change'] = df['change'].replace('Ch', 1)
    df['change'] = df['change'].replace('No', 0)
    df['gender'] = df['gender'].replace('Male', 1)
    df['gender'] = df['gender'].replace('Female', 0)
    df['diabetesMed'] = df['diabetesMed'].replace('Yes', 1)
    df['diabetesMed'] = df['diabetesMed'].replace('No', 0)
    # Value is 0 if not taking the medication
    for col in medications:
        df[col] = df[col].replace('No', 0)
        df[col] = df[col].replace('Steady', 1)
        df[col] = df[col].replace('Up', 1)
        df[col] = df[col].replace('Down', 1)

    # 3 categories; 'None' with value of -99 will be the first category for convenient encoding
    df['A1Cresult'] = df['A1Cresult'].replace('>7', 1)
    df['A1Cresult'] = df['A1Cresult'].replace('>8', 1)
    df['A1Cresult'] = df['A1Cresult'].replace('Norm', 0)
    df['A1Cresult'] = df['A1Cresult'].replace('None', -99)
    df['max_glu_serum'] = df['max_glu_serum'].replace('>200', 1)
    df['max_glu_serum'] = df['max_glu_serum'].replace('>300', 1)
    df['max_glu_serum'] = df['max_glu_serum'].replace('Norm', 0)
    df['max_glu_serum'] = df['max_glu_serum'].replace('None', -99)

    # Encode age. 
    for i in range(0,10):
        df['age'] = df['age'].replace('['+str(10*i)+'-'+str(10*(i+1))+')', i+1)
    df['age'] = df['age'].astype('int64')

    # Encode the prediction label, i.e. whether the patient is readmitted within 30 days
    # The model doesn't perform well if encoding both '>30' and '<30' to 1.
    df['readmitted'] = df['readmitted'].replace('>30', 0)
    df['readmitted'] = df['readmitted'].replace('<30', 1)
    df['readmitted'] = df['readmitted'].replace('NO', 0)

    # Categorize diagnoses into 9 categories, including Circulatory, Respiratory,
    # Digestive, Diabetes, Injury, Musculoskeletal, Genitourinary, Neoplasms, and Others.
    # Ignore secondary and tertiary diagnoses, i.e. diag_2, diag_3
    df['diag'] = df['diag_1']
    df.loc[df['diag_1'].str.contains('V'), ['diag']] = 0
    df.loc[df['diag_1'].str.contains('E'), ['diag']] = 0
    df['diag'] = df['diag'].astype('float')

    for index, row in df.iterrows():
        if (row['diag'] >= 390 and row['diag'] < 460) or (np.floor(row['diag']) == 785):
            df.loc[index, 'diag'] = 1
        elif (row['diag'] >= 460 and row['diag'] < 520) or (np.floor(row['diag']) == 786):
            df.loc[index, 'diag'] = 2
        elif (row['diag'] >= 520 and row['diag'] < 580) or (np.floor(row['diag']) == 787):
            df.loc[index, 'diag'] = 3
        elif (np.floor(row['diag']) == 250):
            df.loc[index, 'diag'] = 4
        elif (row['diag'] >= 800 and row['diag'] < 1000):
            df.loc[index, 'diag'] = 5
        elif (row['diag'] >= 710 and row['diag'] < 740):
            df.loc[index, 'diag'] = 6
        elif (row['diag'] >= 580 and row['diag'] < 630) or (np.floor(row['diag']) == 788):
            df.loc[index, 'diag'] = 7
        elif (row['diag'] >= 140 and row['diag'] < 240):
            df.loc[index, 'diag'] = 8
        else:
            df.loc[index, 'diag'] = 0
    df = df.drop(['diag_1', 'diag_2', 'diag_3'], axis=1)

    # Normalize with log(1 + x)
    # Improve F1 by 0.02
    df['number_outpatient'] = np.log1p(df['number_outpatient'])
    df['number_inpatient'] = np.log1p(df['number_inpatient'])
    df['number_emergency'] = np.log1p(df['number_emergency'])
    
    features = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'glipizide',
         'glyburide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'insulin', 'glyburide-metformin', 
         'A1Cresult', 'change', 'diabetesMed']
    df[features] = df[features].astype('int64')

    # Interaction features; improve F1 by 0.08
    interactionterms = [('num_medications','time_in_hospital'),
    ('num_medications','num_procedures'),
    ('time_in_hospital','num_lab_procedures'),
    ('num_medications','num_lab_procedures'),
    ('num_medications','number_diagnoses'),
    ('age','number_diagnoses'),
    ('change','num_medications'),
    ('number_diagnoses','time_in_hospital'),
    ('num_medications','num_med_change')]
    for inter in interactionterms:
        name = inter[0] + '|' + inter[1]
        df[name] = df[inter[0]] * df[inter[1]]
    
    df['diag'] = df['diag'].astype('object')
    df_pd = pd.get_dummies(df, columns=['gender', 'admission_type_id', 'discharge_disposition_id',
                                    'admission_source_id', 'max_glu_serum', 'A1Cresult', 'diag'], drop_first = True)
    race_feature = pd.get_dummies(df_pd['race'])
    df_pd = pd.concat([df_pd, race_feature], axis=1)      
    df_pd = df_pd.drop(['race'], axis=1)
    print(df_pd.columns)

    # Manually selected features; relief_algorithm func returns error. 
    selected_features = ['age', 'time_in_hospital', 'num_procedures', 'num_medications', 'number_outpatient', 
                 'number_emergency', 'number_inpatient', 'number_diagnoses', 'metformin', 
                 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'glipizide', 'glyburide',
                 'pioglitazone', 'rosiglitazone', 'acarbose', 'insulin', 'glyburide-metformin',
                 'AfricanAmerican', 'Asian', 'Caucasian', 'Hispanic', 'Other', 'gender_1', 
                 'admission_type_id_3', 'admission_type_id_5', 'discharge_disposition_id_2', 'discharge_disposition_id_7', 
                 'discharge_disposition_id_10', 'discharge_disposition_id_18', 'admission_source_id_4',
                 'admission_source_id_7', 'admission_source_id_9', 'max_glu_serum_0', 'max_glu_serum_1', 'A1Cresult_0',
                 'A1Cresult_1', 'num_medications|time_in_hospital', 'num_medications|num_procedures',
                 'time_in_hospital|num_lab_procedures', 'num_medications|num_lab_procedures', 'num_medications|number_diagnoses',
                 'age|number_diagnoses', 'change|num_medications', 'number_diagnoses|time_in_hospital',
                 'num_medications|num_med_change', 'diag_1.0', 'diag_2.0', 'diag_3.0', 'diag_4.0',
                 'diag_5.0','diag_6.0', 'diag_7.0', 'diag_8.0']

    X, y = df_pd[selected_features], df_pd['readmitted']
    # Oversampling due to imbalanced prediction label. Only ~12% of data have label of 1.
    sm = SMOTE(random_state=20)
    X_new, y_new = sm.fit_resample(X.values, y)
    X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=0.3, random_state=42)
     
    return X_train, X_test, y_train, y_test


def create_target(diabetes_df):

    y = (diabetes_df['readmitted'] != 'NO').astype(int).values

    return y


def relief_algorithm(features_df, y):

    data = features_df.values
    
    fs = ReliefF(n_neighbors=5, n_features_to_keep=36)
    X_relieff = fs.fit_transform(data, y)
    top_features = features_df.columns[fs.top_features]

    return top_features


def model_training_random_forest(X_train, y_train, X_test, y_test):

    model = RandomForestClassifier(n_estimators=100, random_state=0,  n_jobs=-1)

    scoring = ['accuracy', 'recall', 'f1']
    cv_scores = cross_validate(model, X_train, y_train, cv=5,
                            scoring=scoring, return_train_score=True, n_jobs=-1)

    # Validation on test set
    model2 = RandomForestClassifier(n_estimators=100, random_state=0,  n_jobs=-1)
    model2.fit(X_train, y_train)
    y_pred_proba = model2.predict(X_test)
    y_pred = (y_pred_proba>0.5).astype(int)
    y_true = y_test
    print("RF")
    print(np.sum(y_true == y_pred) / y_pred.shape[0])
    print (classification_report(y_true, y_pred))

    return cv_scores


def model_training_svm(X_train, y_train, X_test, y_test, n_fold = 5):

    model = SVC(gamma='auto')

    scoring = ['accuracy', 'recall', 'f1']
    cv_scores = cross_validate(model, X_train, y_train, cv=n_fold,
                            scoring=scoring, return_train_score=True, n_jobs=-1)

    # Validation on test set
    model2 = SVC(gamma='auto')
    model2.fit(X_train, y_train)
    y_pred_proba = model2.predict(X_test)
    y_pred = (y_pred_proba>0.5).astype(int)
    y_true = y_test
    print("SVM")
    print(np.sum(y_true == y_pred) / y_pred.shape[0])
    print (classification_report(y_true, y_pred))

    return cv_scores


def plot_cv_scores(cv_scores):

    plt.plot(cv_scores['test_accuracy'], 'o-')
    plt.plot(cv_scores['test_recall'], 'o-')
    plt.plot(cv_scores['test_f1'], 'o-')
    plt.xlabel('K-fold CV Set')
    plt.legend(['Accuracy', 'Recall', 'F1 Score'])

    return


def plot_history(history):

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.xlabel("Epochs")
    plt.legend(['Train Accuracy', 'Val Accuracy'])

    return None


def model_training_cnn(X, y, X_test, y_test):

    X = X.astype('long').reshape(-1, 7, 8)
    # To categorical
    y = to_categorical(y, num_classes=2)
    input_shape=(7, 8, 1)
    num_classes = 2

    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=input_shape),
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same"),
            # 4x 4
            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same"),
            # 2 x 2
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
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
        X, y, epochs=100, batch_size=64, callbacks=callbacks, validation_split=0.3
    )

    # Validation on test set
    X_test = X_test.astype('long').reshape(-1, 7, 8) #101766
    y_test = to_categorical(y_test, num_classes=2)
    y_pred_proba = model.predict(X_test)[:, 1]
    y_pred = (y_pred_proba>0.5).astype(int)
    y_true = y_test[:, 1]
    
    print (classification_report(y_true, y_pred))

    return history


def get_intermediate_layer_output(X):

    model = tf.keras.models.load_model('models/model.hdf5')

    aux_model = tf.keras.Model(inputs=model.inputs,
                           outputs=model.outputs + [model.layers[6].output])

    # Access both the final and intermediate output of the original model
    final_output, X_embedding = aux_model.predict(X)

    return X_embedding