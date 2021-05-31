import pandas as pd
import numpy as np
from Packages.data import load_csv_to_dataframe, plot_roc_curve, plot_feature_importance # show_tree,
from Packages.data import plot_learning_curves, train_test
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, mean_squared_error

file = '../Data/data.csv'
df = load_csv_to_dataframe(file)

X = df.drop(['customerID', 'tenure', 'MonthlyCharges', 'TotalCharges', 'Churn', 'Churn_cat'], axis=1)
X = pd.get_dummies(X, drop_first=True)
y = df['Churn_cat']

X_train, X_test, y_train, y_test = train_test(X, y)

tree_clf = RandomForestClassifier(n_estimators=100)
tree_clf.fit(X_train, y_train)
y_predict_test = tree_clf.predict(X_test)
y_predict_train = tree_clf.predict(X_train)

# Grid Search CV
params = {'max_depth': [2, 5, 10, 20],
          'min_samples_split': [3, 10, 30],
          'max_leaf_nodes': [3, 10, 30],
          'max_features': [3, 10, 30],
          'min_samples_leaf': [100, 102, 104, 106]}

grid_search_cv = GridSearchCV(RandomForestClassifier(random_state=42), params, cv=5)
grid_search_cv.fit(X_train, y_train)
best_params = grid_search_cv.best_params_
print(f"Best params: {grid_search_cv.best_params_}")

# Grid Search CV: best parameters for RFC
rfc_clf_bp = RandomForestClassifier(**best_params)
rfc_clf_bp.fit(X_train, y_train)
y_predict_bp = rfc_clf_bp.predict(X_test)
y_train_predict_bp = rfc_clf_bp.predict(X_train)
bp_score = rfc_clf_bp.score(X_test, y_test)


def display_scores(scores):
    print(f"Scores: {scores}")
    print(f"Mean: {scores.mean():.4f}")
    print(f"Std: {scores.std():.4f}")


# Cross-Val on tree_clf : no parameters
y_predict_cv = cross_val_predict(tree_clf, X_train, y_train, cv=5)
y_cv_scores = cross_val_score(tree_clf, X_train, y_train, cv=5)
print(f"Cross-Val predictions score: {y_predict_cv}")
display_scores(y_cv_scores)

# Cross-Val on tree_clf_bp : best parameters
y_predict_bp_cv = cross_val_predict(rfc_clf_bp, X_train, y_train, cv=5)
y_bp_cv_scores = cross_val_score(rfc_clf_bp, X_train, y_train, cv=5, scoring='accuracy')
print(f"Cross-Val predictions score: {y_predict_bp_cv}")
display_scores(y_bp_cv_scores)

# Confusion Matrix results
print("Untuned, training data")
print(confusion_matrix(y_train, y_predict_train))
print("Untuned, test data")
print(confusion_matrix(y_test, y_predict_test))
print()
print("Best parameters, test data")
print(confusion_matrix(y_test, y_predict_bp))

print()
print("Untuned: Cross-Val, training data")
print(confusion_matrix(y_train, y_predict_cv))
print()
print("Best parameters: Cross-Val, training data")
print(confusion_matrix(y_train, y_predict_bp_cv))

print(sum(y_predict_train))
print(sum(y_predict_cv))
print(sum(y_predict_bp_cv))
################################################33
print("Accuracy for untuned")
print(f"Accuracy on training set: {tree_clf.score(X_train, y_train):.4f}")
print(f"Accuracy on test set:     {tree_clf.score(X_test, y_test):.4f}")
print(f"Accuracy on cv set:       {tree_clf.score(X_train, y_predict_cv):.4f}")

print("Accuracy for tuned")
print(f"Accuracy on training set: {rfc_clf_bp.score(X_train, y_train):.4f}")
print(f"Accuracy on test set:     {rfc_clf_bp.score(X_test, y_test):.4f}")
print(f"Accuracy on cv set:       {rfc_clf_bp.score(X_train, y_predict_bp_cv):.4f}")

######################################################
print()
# Precision, Recall and F1 Scores
print("Precision, Recall, F1 scores for 'Untuned'")
print("="*42)
print(f"Precision (test) : {precision_score(y_test, y_predict_test):.4f}")
print(f"Recall    (test) : {recall_score(y_test, y_predict_test):.4f}")
print(f"F1        (test) : {f1_score(y_test, y_predict_test):.4f}")
print('*'*8 + "Cross-Validation "+'*'*8)
print(f"Precision (train): {precision_score(y_train, y_predict_cv):.4f}")
print(f"Recall    (train): {recall_score(y_train, y_predict_cv):.4f}")
print(f"F1        (train): {f1_score(y_train, y_predict_cv):.4f}")

print("Precision, Recall, F1 scores for 'Best Parameters'")
print("="*42)
print(f"Precision (test) : {precision_score(y_test, y_predict_bp):.4f}")
print(f"Recall    (test) : {recall_score(y_test, y_predict_bp):.4f}")
print(f"F1        (test) : {f1_score(y_test, y_predict_bp):.4f}")
print('*'*8 + "Cross-Validation "+'*'*8)
print(f"Precision (train): {precision_score(y_train, y_predict_bp_cv):.4f}")
print(f"Recall    (train): {recall_score(y_train, y_predict_bp_cv):.4f}")
print(f"F1        (train): {f1_score(y_train, y_predict_bp_cv):.4f}")

# ROC Curve
# Compute y scores for input to metrics
y_probas = cross_val_predict(rfc_clf_bp, X_train, y_train, cv=5, method='predict_proba')  # method='decision_function'
y_scores = y_probas[:, 1]
fpr, tpr, thresholds = roc_curve(y_train, y_scores)
#plot_roc_curve(fpr, tpr)

# ROC AUC Curve
print(f"ROC AUC: {roc_auc_score(y_train, y_scores):.4f}")

# print(f"Feature importances:\n{tree_clf.feature_importances_}")

#plot_feature_importance(X, rfc_clf_bp)

#plot_learning_curves(rfc_clf_bp, X, y, 1000)

# print(f"Accuracy on training set: {tree_clf.score(X_train, y_train):.4f}")  # ? not fitted yet ??
print(f"Accuracy on test set: {rfc_clf_bp.score(X_test, y_test):.4f}")

# Classification Report Comparison: DT, CV and GSCV

print("=====================  y_test -- y_predict ============")
print("="*55)
print(classification_report(y_test, y_predict_test))
print("="*55)
print("======  y_test -- y_predict_best_parameters ==")
print("="*55)
print(classification_report(y_test, y_predict_bp))

# # Create an instance of every classifier for comparison
# knn_clf = KNeighborsClassifier(7)
# log_clf = LogisticRegression(max_iter=1000)
# lsvm_clf = SVC(kernel="linear", C=0.025)
# rbfsvm_clf = SVC(gamma=2, C=1)
# gaus_clf = GaussianProcessClassifier(1.0 * RBF(1.0))
# rf_clf = RandomForestClassifier(**best_params)
# mlpnn_clf = MLPClassifier(max_iter=1000)
# ada_clf = AdaBoostClassifier()
# gausb_clf = GaussianNB()
# qda_clf = QuadraticDiscriminantAnalysis()
#
# voting_clf = VotingClassifier(
#     estimators=[('knn', knn_clf),
#                 ('log', log_clf),
#                 ('lsvm', lsvm_clf),
#                 ('gaus', gaus_clf),
#                 ('rf', rf_clf),
#                 ('ada', ada_clf),
#                 ],
#     voting='hard')
# voting_clf.fit(X_train, y_train)
# test_acc = []
# for clf in (knn_clf, log_clf, lsvm_clf, gaus_clf, rf_clf, ada_clf, voting_clf): # rbfsvm_clf,gausb_clf, qda_clf,mlpnn_clf,
#     clf.fit(X_train, y_train)
#     y_pred = clf.predict(X_test)
#     test_acc.append((clf.__class__.__name__, accuracy_score(y_test, y_pred)))
# print(test_acc)
# Bagging Decision Tree Classifier
bag_clf = BaggingClassifier(
    RandomForestClassifier(**best_params), n_estimators=500,
    max_samples=100, bootstrap=True, n_jobs=-1)
bag_clf.fit(X_train, y_train)

y_pred_bag_dt = bag_clf.predict(X_test)# Bagging Logistic Regression Classifier
bag_clf = BaggingClassifier(
    LogisticRegression(max_iter=2000), n_estimators=500,
    max_samples=100, bootstrap=True, n_jobs=-1)
bag_clf.fit(X_train, y_train)

y_pred_bag_lr = bag_clf.predict(X_test)

# bagging
print(classification_report(y_test, y_pred_bag_dt))

print(classification_report(y_test, y_pred_bag_lr))

adaboost_clf = AdaBoostClassifier(
    base_estimator=rfc_clf_bp, n_estimators=10,
    algorithm="SAMME.R", learning_rate= 1)
# Accuracy: 0.8599857853589197

adaboost_clf.fit(X_train, y_train)

# Generate predictions for test dataset
y_pred_boost = adaboost_clf.predict(X_test)
# Boosting
print("Accuracy:", accuracy_score(y_test, y_pred_boost))
print(f"Classification Report: {classification_report(y_test, y_pred_boost)}")

# l2 (ridge) penalty

best_model_task2 = LogisticRegressionCV(
    # Each of the values in Cs describes the inverse of regularization strength.
    Cs=list(np.power(10.0, np.arange(-10, 10))),
    # penalty{‘l1’, ‘l2’, ‘elasticnet’}, default=’l2’
    # The ‘newton-cg’, ‘sag’ and ‘lbfgs’ solvers support only l2 penalties.
    # ‘elasticnet’ is only supported by the ‘saga’ solver.
    penalty='l2', #'l1', #
    # The default scoring option used is ‘accuracy’.
    # For a list of scoring functions that can be used, look at sklearn.metrics
    scoring='roc_auc',# 'accuracy',#'neg_log_loss',#
    cv=5,
    random_state=777,
    max_iter=10000,
    # Specifies if a constant (a.k.a. bias or intercept) should be added to the decision function.
    # fit_intercept : bool, default=True
    fit_intercept=True,
    # Algorithm to use in the optimization problem.
    # solver{‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}, default=’lbfgs’
    solver='newton-cg',#'lbfgs', # 'sag',#
    # l1_ratioslist of float, default=None
    #l1_ratios = [0.5],
    # Tolerance for stopping criteria
    tol=10
)
best_model_task2.fit(X_train, y_train)

print('Max auc_roc:', best_model_task2.scores_[1].max())
print(f"Accuracy score on train data: {best_model_task2.score(X_train, y_train):.4f}")
print(f"Accuracy score on test data: {best_model_task2.score(X_test, y_test):.4f}")

