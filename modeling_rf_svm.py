from util import *

# Whether to generate training data based on domain knowledge
ENABLE_FEATURE_ENGINEERING = True

if ENABLE_FEATURE_ENGINEERING:
	X_train, X_test, y_train, y_test = generate_training_data_with_feature_engineering()
else:
	X_train, X_test, y_train, y_test = generate_training_data()

rf_cv_scores = model_training_random_forest(X_train, y_train, X_test, y_test)
plot_cv_scores(rf_cv_scores, "Random Forest")

svm_cv_scores = model_training_svm(X_train, y_train, X_test, y_test)
plot_cv_scores(svm_cv_scores, "SVM")