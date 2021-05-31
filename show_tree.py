import pandas as pd
import imageio
import io
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from Packages.data import load_csv_to_dataframe
import pydotplus  # issue with Anaconda


def show_tree(tree_classifier, features, path):
    f = io.StringIO()
    export_graphviz(tree_classifier, out_file=f, feature_names=features)
    pydotplus.graph_from_dot_data(f.getvalue()).write_png(path)
    img = imageio.imread(path)
    plt.rcParams["figure.figsize"] = (20, 20)
    plt.imshow(img)


# Load data
file = '../Data/data.csv'
df = load_csv_to_dataframe(file)
# print(data.head())

# Prepare dataframe for train test split
X = df.drop(['customerID', 'tenure', 'MonthlyCharges', 'TotalCharges', 'Churn', 'Churn_cat'], axis=1)
X = pd.get_dummies(X, drop_first=True)
y = df['Churn_cat']
# y = pd.get_dummies(y, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print(f"training shape: {X_train.shape}, testing shape: {X_test.shape}")
print(f"training labels: {y_train.shape}, testing labels: {y_test.shape}")

# Run fit and predict for classifier
tree_clf = DecisionTreeClassifier(max_depth=5, max_features=30, max_leaf_nodes=10, min_samples_leaf=100, min_samples_split=3)
tree_clf.fit(X_train, y_train)
y_predict = tree_clf.predict(X_test)

print(f"Accuracy on training set: {tree_clf.score(X_train, y_train):.4f}")
print(f"Accuracy on test set: {tree_clf.score(X_test, y_test):.4f}")


# Plot of Decision Tree
feature_cols = X.columns

show_tree(tree_clf, feature_cols, './Images/tree.png')