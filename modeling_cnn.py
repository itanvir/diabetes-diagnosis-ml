from util import *

# Whether to generate training data based on domain knowledge
ENABLE_FEATURE_ENGINEERING = True

if ENABLE_FEATURE_ENGINEERING:
	X_train, X_test, y_train, y_test = generate_training_data_with_feature_engineering()
	model_checkpoint_path = "models/model_with_fe.hdf5"
	input_shape = (7, 8)
else:
	X_train, X_test, y_train, y_test = generate_training_data()
	model_checkpoint_path = "models/model.hdf5"
	input_shape = (6, 6)

history = model_training_cnn(X_train, y_train, X_test, y_test, model_checkpoint_path, input_shape)
plot_history(history)

X_train_embedding = get_intermediate_layer_output(X_train, model_checkpoint_path, input_shape)
X_test_embedding = get_intermediate_layer_output(X_test, model_checkpoint_path, input_shape)

rf_cv_scores = model_training_random_forest(X_train_embedding, y_train, X_test_embedding, y_test)
plot_cv_scores(rf_cv_scores, "CNN-RF")

svm_cv_scores = model_training_svm(X_train_embedding, y_train, X_test_embedding, y_test)
plot_cv_scores(svm_cv_scores, "CNN-SVM")