import pandas as pd

# import sklearn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# import matplotlib
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# read wine dataset from csv file (from Kaggle : https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009/data)
wine_data = pd.read_csv('winequality-red.csv')

# check if there's any missing values
print(wine_data.isnull().sum())
# result : no missing values

# check the columns type
print(wine_data.dtypes)
# result : all columns are numerical (int64 for quality and float64 for the rest)


# define features (X) and target (y)
X = wine_data.drop('quality', axis=1)
y = wine_data['quality']

# check the disctinct labels values
print(wine_data['quality'].unique())
# result : [5 6 7 4 8 3]

# split the dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# set a maximum depth for the decision tree
max_depth = 4

# create a decision tree classifier
clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)

# train the classifier
clf.fit(X_train, y_train)

# make predictions on the test set
y_pred = clf.predict(X_test)

# calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'accuracy = {accuracy}')

multiclass_report = classification_report(y_test, y_pred)
print(f'multiclass_report = {multiclass_report}')
conf_matrix = confusion_matrix(y_test, y_pred)
# seems like the tree accuracy is very variable between each class

# display the tree
fig, ax = plt.subplots(figsize=(40, 20))
plot_tree(clf, feature_names=X.columns, class_names=wine_data['quality'].unique().astype(str), filled=True, ax=ax)
plt.title("Decision tree with original labels")
plt.show()

############################################################
# same classification but with binary labels (high/low quality)

y2 = (wine_data['quality'] >= 5).astype(int)  # 0 for low quality, 1 for high

# split the dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y2, test_size=0.2, random_state=42)

# create a decision tree classifier
clf2 = DecisionTreeClassifier(max_depth=max_depth, random_state=42)

# train the classifier
clf2.fit(X_train, y_train)

# make predictions on the test set
y_pred = clf2.predict(X_test)

# calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'accuracy = {accuracy}')

binary_class_report2 = classification_report(y_test, y_pred)
print(f'binary_class_report2 = {binary_class_report2}')
conf_matrix2 = confusion_matrix(y_test, y_pred)

# display the tree
fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(clf2, feature_names=X.columns, class_names=y2.unique().astype(str), filled=True, ax=ax)
plt.title("Decision tree with binary labels")
plt.show()