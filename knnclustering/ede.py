import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('pattern.csv')
df.head()

df.iloc[59]

# remove outlier(s)
df2 = df.drop( [df.index[59] ] )
df2.iloc[59]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.set_title("Dataset with outlier")
sns.scatterplot(ax=ax1, data=df, x='X1', y='X2', hue='GR', palette='winter')

ax1.set_title("Dataset with outlier removed")
sns.scatterplot(ax=ax2, data=df2, x='X1', y='X2', hue='GR', palette='winter')

plt.savefig('scatter.jpg')

# Model fitting

from sklearn.cluster import KMeans

X = df2[['X1', 'X2']]
y = df2[['GR']]

kmeans = KMeans(n_clusters=3)
kmeans = kmeans.fit(X)
# Getting the cluster labels
labels = kmeans.predict(X)
# Centroid values
centroids = kmeans.cluster_centers_

print(centroids)

# Re-organise labels so that predictions can match original labels
actual = y
actual['GR'] = np.where(actual['GR']==2, 0, actual['GR'])
actual['GR'] = np.where(actual['GR']==3, 2, actual['GR'])

from sklearn.metrics import classification_report

target_names = ['GR 1', 'GR 2', 'GR 3']

print(classification_report(actual, labels, target_names=target_names))

from sklearn.metrics import confusion_matrix

confusion_matrix(actual, labels)

actual.value_counts()

df2['predictions'] = labels

plt.figure(figsize=(10, 6))
plt.title('K-means clustering solution')
sns.scatterplot(data=df2, x='X1', y='X2', hue='predictions', palette='viridis')

plt.legend(labels=['GR 1', 'GR 2', 'GR 3'])
plt.savefig('kmeansolution.jpg')

from sklearn.metrics import classification_report

target_names = ['Class 1', 'Class 2', 'Class 3']

print(classification_report(df.actual, df.predicted, target_names=target_names))
