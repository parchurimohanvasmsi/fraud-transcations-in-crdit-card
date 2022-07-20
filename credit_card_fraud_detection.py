import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import sklearn
from pylab import rcParams
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

rcParams['figure.figsize'] = 14, 8
RANDOM_SEED = 42
LABELS = ["Normal", "Fraud"]

"""# Data Exploration"""

data = pd.read_csv("creditcard.csv")

print('Dataset contains {} rows and {} columns.'.format(data.shape[0], data.shape[1]))

data.head()

data.sample(5)

data.info()

data.isnull().values.any()

count_classes = pd.value_counts(data['Class'], sort = True)

print(count_classes)

count_classes.plot(kind = 'bar', rot=0)

plt.xticks(range(2), LABELS)

plt.title("Transaction Class Distribution")

plt.xlabel("Class")

plt.ylabel("Frequency")

fraud  = data[data['Class']==1]

normal = data[data['Class'] == 0]

fraud.Amount.describe()

normal.Amount.describe()

"""Let us take a look at the distribution of 'Time' and 'Amount' features, as they are the only non PCA transformed features"""

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Amount per transaction by class')
bins = 50
ax1.hist(fraud.Amount, bins = 5,color="Orange")
ax1.set_title('Fraud')
ax2.hist(normal.Amount, bins = bins)
ax2.set_title('Normal')
plt.xlabel('Amount ($)')
plt.ylabel('Number of Transactions')
plt.xlim((0, 20000))
plt.yscale('log')
plt.show();

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Time of transaction vs Amount by class')
ax1.scatter(fraud.Time, fraud.Amount,color='Orange')
ax1.set_title('Fraud')
ax2.scatter(normal.Time, normal.Amount)
ax2.set_title('Normal')
plt.xlabel('Time (in Seconds)')
plt.ylabel('Amount')
plt.show()

#finding correlation between columns and plotting heatmap

plt.figure(figsize=(12,10))
g=sns.heatmap(data.corr(),annot=False,cmap="seismic")
plt.show()

"""# Data Cleaning

## *Feature Scaling*

Anonymised features appear to have been scaled and centred around zero but Amount and Time have not been scaled. For algorithms to perform well, the features have to be scaled.
"""

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler2 = StandardScaler()

#scaling Time feature

scaled_time = scaler.fit_transform(data[['Time']])
flat_list1 = [item for sublist in scaled_time.tolist() for item in sublist]    # making a flat list out of a list of lists
scaled_time = pd.Series(flat_list1)

#scaling Amount feature

scaled_amount = scaler2.fit_transform(data[['Amount']])
flat_list2 = [item for sublist in scaled_amount.tolist() for item in sublist]
scaled_amount = pd.Series(flat_list2)

# adding the scaled features and dropping the unscaled features from the original dataframe

df = pd.concat([data, scaled_amount.rename('scaled_amount'), scaled_time.rename('scaled_time')], axis=1)
df.drop(['Amount', 'Time'], axis=1, inplace=True)
df.sample(5)

"""## *Data splitting*

The dataset is now split into training set and testing set for further processes.
"""

from sklearn.model_selection import train_test_split

X = data.iloc[:,:-1]
Y= data['Class']


X_pretrain, X_test, Y_pretrain, Y_test = train_test_split(X, Y, test_size = 0.35)

print("Train and test sizes, respectively:", len(X_pretrain), len(Y_pretrain), "|", len(X_test), len(Y_test))

print("Total number of frauds:", len(Y.loc[data['Class'] == 1]), len(Y.loc[data['Class'] == 1])/len(Y))

print("Number of frauds on y_test:", len(Y_test.loc[data['Class'] == 1]), len(Y_test.loc[data['Class'] == 1]) / len(Y_test))

print("Number of frauds on y_pretrain:", len(Y_pretrain.loc[data['Class'] == 1]), len(Y_pretrain.loc[data['Class'] == 1])/len(Y_pretrain))

"""## *SMOTE oversampling*

The present dataset is highly imbalanced with the fraud cases forming an extremely small percentage ( < 1% ) of the whole. This poses a problem in training the algorithms. To deal with this problem, the technique of SMOTE oversampling is used.
"""

from imblearn.over_sampling import SMOTE

X_train, Y_train = SMOTE().fit_resample(X_pretrain, Y_pretrain)

pd.value_counts(Y_train, sort = True).plot(kind = 'bar', rot=0)
plt.xticks(range(2), LABELS)
plt.title("Transaction Class Distribution")
plt.xlabel("Class")
plt.ylabel("Frequency")

"""# Model Training

## 1. Simple Logistic Regression
"""

from sklearn import datasets, linear_model
from sklearn.metrics import confusion_matrix

logistic = linear_model.LogisticRegression(max_iter = 500)
logistic.fit(X_train, Y_train)

Y_predicted = np.array(logistic.predict(X_test))
Y_right = np.array(Y_test)
g = confusion_matrix(Y_right,Y_predicted)
g

plt.figure(figsize=(5,5))
sns.heatmap(g,linewidths=.5,annot=True,cmap="warmcool")

false_pos = g[0][1]

false_neg = g[1][0]

false_neg_rate = false_neg/(false_pos+false_neg)

accuracy = (len(X_test) - (false_neg + false_pos)) / len(X_test)

print("Accuracy:", accuracy)
print("False negative rate (with respect to misclassifications): ", false_neg_rate)
print("False negative rate (with respect to all the data): ", false_neg / len(X_test))
print("False negatives, false positives, mispredictions:", false_neg, false_pos, false_neg + false_pos)
print("Total test data points:", len(X_test))

"""## 2. K NN"""

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

X_scaled = scale(X)

pca = PCA(n_components=2)

X_reduced = pca.fit_transform(X_scaled)

X_train2, X_test2, Y_train2, Y_test2 = train_test_split(X_reduced, Y, test_size = 0.33, random_state=500)

kmeans = KMeans(init = 'k-means++', n_clusters=2,n_init=10)

kmeans.fit(X_train2)

Y_predicted2 = np.array(kmeans.predict(X_test2))

Y_right2 = np.array(Y_test2)

g2 = confusion_matrix(Y_right2, Y_predicted2)

g2

plt.figure(figsize=(5,5))

sns.heatmap(g2,linewidths=.5,annot=True,cmap="cool")

h = .01     # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = X_reduced[:, 0].min() - 1, X_reduced[:, 0].max() + 1
y_min, y_max = X_reduced[:, 1].min() - 1, X_reduced[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap="winter",
           aspect='auto', origin='lower')

plt.plot(X_reduced[:, 0], X_reduced[:, 1], 'k.', markersize=2)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('K-means clustering on the credit card fraud dataset (PCA-reduced data)\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()

false_pos = g2[0][1]

false_neg = g2[1][0]

false_neg_rate = false_neg/(false_pos+false_neg)

accuracy = (len(X_test2) - (false_neg + false_pos)) / len(X_test2)

print("Accuracy:", accuracy)
print("False negative rate (with respect to misclassifications): ", false_neg_rate)
print("False negative rate (with respect to all the data): ", false_neg / len(X_test2))
print("False negatives, false positives, mispredictions:", false_neg, false_pos, false_neg + false_pos)
print("Total test data points:", len(X_test2))

"""## 3. Random Forest"""

from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=20,criterion='entropy', random_state=0,max_depth=10)
random_forest.fit(X_train, Y_train)

Y_predicted3 = np.array(random_forest.predict(X_test))

Y_right3 = np.array(Y_test)

g3 = confusion_matrix(Y_right3, Y_predicted3)

g3

plt.figure(figsize=(5,5))

sns.heatmap(g3,linewidths=.5,annot=True,cmap="cool")

false_pos = g3[0][1]

false_neg = g3[1][0]

false_neg_rate = false_neg/(false_pos+false_neg)

accuracy = (len(X_test) - (false_neg + false_pos)) / len(X_test)

print("Accuracy:", accuracy)
print("False negative rate (with respect to misclassifications): ", false_neg_rate)
print("False negative rate (with respect to all the data): ", false_neg / len(X_test))
print("False negatives, false positives, mispredictions:", false_neg, false_pos, false_neg + false_pos)
print("Total test data points:", len(X_test))

"""## 4. SVM"""

from sklearn.svm import LinearSVC

svc_model = LinearSVC()
svc_model.fit(X_train, Y_train)

Y_predicted5 = np.array(svc_model.predict(X_test))

Y_right5 = np.array(Y_test)

g5 = confusion_matrix(Y_right5, Y_predicted5)

g5

plt.figure(figsize=(5,5))

sns.heatmap(g5,linewidths=.5,annot=True,cmap="coolwarm,12")

false_pos = g5[0][1]

false_neg = g5[1][0]

false_neg_rate = false_neg/(false_pos+false_neg)

accuracy = (len(X_test) - (false_neg + false_pos)) / len(X_test)

print("Accuracy:", accuracy)
print("False negative rate (with respect to misclassifications): ", false_neg_rate)
print("False negative rate (with respect to all the data): ", false_neg / len(X_test))
print("False negatives, false positives, mispredictions:", false_neg, false_pos, false_neg + false_pos)
print("Total test data points:", len(X_test))






