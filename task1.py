import pandas as pd
import numpy as np
from Packages.data import load_csv_to_dataframe, plot_roc_curve, plot_feature_importance  # show_tree,
from Packages.data import plot_learning_curves, train_test
from sklearn.model_selection import cross_val_predict, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.base import BaseEstimator


class NeverChurnClassifier(BaseEstimator):
    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)


def display_scores(scores):
    print(f"Scores: {scores}")
    print(f"Mean: {scores.mean():.4f}")
    print(f"Std: {scores.std():.4f}")


# Import data and load to dataframe
file = '../Data/data.csv'
df = load_csv_to_dataframe(file)
X = df.drop(['customerID', 'tenure', 'MonthlyCharges', 'TotalCharges', 'Churn', 'Churn_cat'], axis=1)
X = pd.get_dummies(X, drop_first=True)
y = df['Churn_cat']
X_train, X_test, y_train, y_test = train_test(X, y)

# Create basic DTC model and make predictions
tree_clf = DecisionTreeClassifier()
tree_clf.fit(X_train, y_train)
y_predict = tree_clf.predict(X_test)

# Use Grid Search CV to find 'best' parameters

params = {'max_depth': [2, 5, 10, 20],
          'min_samples_split': [3, 10, 30],
          'max_leaf_nodes': [3, 10, 30],
          'max_features': [3, 10, 30],
          'min_samples_leaf': [100, 102, 104, 106]}
#
grid_search_cv = GridSearchCV(DecisionTreeClassifier(random_state=42), params, cv=5)
grid_search_cv.fit(X_train, y_train)
best_params = grid_search_cv.best_params_

# Use 'best parameters' from GridSearchCV to create new model

best_model_task1 = DecisionTreeClassifier(**best_params)
best_model_task1.fit(X_train, y_train)
y_predict_bp = best_model_task1.predict(X_test)
bp_score = best_model_task1.score(X_test, y_test)

# Cross-Val on tree_clf : no parameters
y_predict_cv = cross_val_predict(tree_clf, X_train, y_train, cv=5)
y_cv_scores = cross_val_score(tree_clf, X_train, y_train, cv=5)

# Cross-Val on best_model_task1 : best parameters
y_predict_bp_cv = cross_val_predict(best_model_task1, X_train, y_train, cv=5)
y_bp_cv_scores = cross_val_score(best_model_task1, X_train, y_train, cv=5, scoring='accuracy')

# print(f"{display_scores(y_cv_scores)}")
print(f"Best params: {grid_search_cv.best_params_}")
print(f"{display_scores(y_bp_cv_scores)}")
print("Best parameters: test labels vs prediction labels")
print(confusion_matrix(y_test, y_predict_bp))
print("Best parameters: training labels vs Cross-Validation labels")
print(confusion_matrix(y_train, y_predict_bp_cv))
print()
print("Accuracy for 'Best Parameters' decision tree")
print(f"Accuracy on training:   {best_model_task1.score(X_train, y_train):.4f}")
print(f"Accuracy on test:       {best_model_task1.score(X_test, y_test):.4f}")
print(f"Accuracy on cv=5:       {best_model_task1.score(X_train, y_predict_cv):.4f}")
print()
print("Precision, Recall, F1 scores for 'Best Parameters'")
print("=" * 42)
print(f"Precision: {precision_score(y_test, y_predict_bp):.4f}")
print(f"Recall:    {recall_score(y_test, y_predict_bp):.4f}")
print(f"F1:        {f1_score(y_test, y_predict_bp):.4f}")
print('*' * 8 + "Cross-Validation " + '*' * 8)
print(f"Precision: {precision_score(y_train, y_predict_bp_cv):.4f}")
print(f"Recall:    {recall_score(y_train, y_predict_bp_cv):.4f}")
print(f"F1:        {f1_score(y_train, y_predict_bp_cv):.4f}")
print()
print("=====  y_test -- y_predict_best_parameters  =====")
print("=" * 55)
print(classification_report(y_test, y_predict_bp))

# ROC Curve
# Compute y scores for input to metrics
y_probas = cross_val_predict(best_model_task1, X_train, y_train, cv=5,
                             method='predict_proba')  # method='decision_function'
y_scores = y_probas[:, 1]
fpr, tpr, thresholds = roc_curve(y_train, y_scores)
plot_roc_curve(fpr, tpr)

# ROC AUC Curve
print(f"ROC AUC: {roc_auc_score(y_train, y_scores):.4f}")

# plot_feature_importance(X, best_model_task1)
