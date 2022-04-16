import numpy as np
import pandas as pd


pd.set_option('display.max_columns', None)
plt.style.use('ggplot')

from util import *


diabetes_df = read_diabetes_data()
features_df = data_cleaning(diabetes_df)
features_df = label_encoding(features_df)
y = create_target(diabetes_df)
top_features = relief_algorithm(features_df, y)

# Model training random forest/all features
X = features_df[top_features[0:36]]
cv_scores = model_training_random_forest(X, y)

print (cv_scores)