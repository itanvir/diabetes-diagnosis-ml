from util import *

df = read_diabetes_data()

n_rows = df.shape[0]
labels = ['<30', '>30', 'NO']
label_pct = []
for label in labels:
	count = df[df['readmitted'] == label]['readmitted'].count()
	label_pct.append(count / n_rows * 100)
	
plt.figure()
plt.bar(labels,label_pct)
plt.xlabel("Readmitted")
plt.ylabel('Percentage %')
plt.savefig("Target Label Distribution")

df = df.drop(['readmitted'], axis = 1)
features = df.columns
primary_value_pct = []
for feature in features:
	count = df.groupby(feature).size().max()
	primary_value_pct.append(count / n_rows * 100)

plt.figure()
plt.bar(range(len(features)), primary_value_pct)
plt.xlabel("Feature IDs")
plt.ylabel('Primary Value %')
plt.savefig("Primary Feature Value Percentage")