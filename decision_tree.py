import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, roc_auc_score, precision_recall_curve
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from Packages.data import load_csv_to_dataframe, show_tree, plot_roc_curve, plot_feature_importance

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
tree_clf = DecisionTreeClassifier(max_depth=10, min_samples_split=100)
tree_clf.fit(X_train, y_train)
y_predict = tree_clf.predict(X_test)

print(f"Accuracy on training set: {tree_clf.score(X_train, y_train):.4f}")
print(f"Accuracy on test set: {tree_clf.score(X_test, y_test):.4f}")

# Cross-Validation

y_predict_cv = cross_val_predict(tree_clf, X_train, y_train, cv=3)

# Confusion Matrix results
print(confusion_matrix(y_test, y_predict))
print(confusion_matrix(y_train, y_predict_cv))

# Precision, Recall and F1 Scores

print(f"Precision (cv): {precision_score(y_train, y_predict_cv):.4f}")
print(f"Recall (cv): {recall_score(y_train, y_predict_cv):.4f}")
print(f"Precision: {precision_score(y_test, y_predict):.4f}")
print(f"Recall: {recall_score(y_test, y_predict):.4f}")
print(f"F1 (cv): {f1_score(y_train, y_predict_cv):.4f}")
print(f"F1: {f1_score(y_test, y_predict):.4f}")

# Compute y scores for input to metrics
y_probas = cross_val_predict(tree_clf, X_train, y_train, cv=3, method='predict_proba')  # method='decision_function'
y_scores = y_probas[:, 1]

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_train, y_scores)
plot_roc_curve(fpr, tpr)

# ROC AUC Curve
print(f"ROC AUC: {roc_auc_score(y_train, y_scores):.4f}")

# Feature Importance
print(f"Feature importances:\n{tree_clf.feature_importances_}")

plot_feature_importance(X, tree_clf)

# Grid Search CV
from sklearn.model_selection import GridSearchCV
params = {'max_leaf_nodes': list(range(2,100)),
          'min_samples_split': [2, 3, 4]}
grid_search_cv = GridSearchCV(DecisionTreeClassifier(random_state=42), params, verbose=1, cv=3)
grid_search_cv.fit(X_train, y_train)
print(f"Grid search CV, best estimator: {grid_search_cv.best_estimator_}")


# # Precision - Recall Trade-off (for SGDClassifier)

# precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores)
#
#
# def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
#     plt.plot(thresholds, precisions[:,-1], "b--", label="Precision")
#     plt.plot(thresholds, recalls[:,-1], "g-", label="Recalls")
#
#
# plot_precision_recall_vs_threshold(precisions, recalls, thresholds)



# # Confusion Matrix plot
# matrix = confusion_matrix(y_test, y_predict)
# # create pandas dataframe
# class_names = ['Churn_no', 'Churn_yes']
# dataframe_Confusion = pd.DataFrame(matrix, index=class_names, columns=class_names)
# # create heatmap
# sns.heatmap(dataframe_Confusion, annot=True,  cmap="Blues", fmt=".0f")
# plt.title("Confusion Matrix")
# plt.tight_layout()
# plt.ylabel("True Class")
# plt.xlabel("Predicted Class")
# plt.savefig('./Images/confusion_matrix.png')
# # plt.show()
#
# Plot of Decision Tree
feature_cols = X.columns

# pydotplus.graphviz.InvocationException: GraphViz's executables not found
# works in Linux after sudo apt-get install graphviz
# for Win10, might have to edit Environment variable ???
show_tree(tree_clf, feature_cols, './Images/tree.png')
