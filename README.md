# diabetes-diagnosis-ml
He et al.(2019) developed five machine learning/deep learning models from the diagnostic records to predict whether the diabetic patients need to be readmitted to the hospital. This project reproduces all the five models, including SVM, Random Forest, CNN, combined CNN-SVM, and combined CNN-RF. The data set were archived by UCI (https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008).  

This project supports two different approaches to encode the features. Users can select the approach with the "ENABLE_FEATURE_ENGINEERING" variable in modeling_rf_svm.py and modeling_cnn.py files.

Dependencies
- matplotlib
- numpy
- pandas
- sklearn
- tensorflow
- keras
- ReliefF
- imblearn
- jupyterlab

Commands to run the code:
- plot data statistics: $ python data_stats.py
- train SVM and RF models: $ python modeling_rf_svm.py
- train CNN, CNN-SVM, and CNN-RF models: $ python modeling_cnn.py
