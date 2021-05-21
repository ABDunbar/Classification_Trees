import pandas as pd
import numpy as np
import io
from sklearn.tree import DecisionTreeClassifier, export_graphviz
# import pydotplus # issue with Anaconda
# from scipy import misc  # imageio replaces
import imageio
import matplotlib.pyplot as plt


def load_csv_to_dataframe(file):
    df = pd.read_csv(file)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['Churn_cat'] = (df['Churn'] == 'Yes').astype(int)
    df['tenure_cat'] = pd.cut(df['tenure'],
                              bins=[0, 10, 20, 30, 40, 50, 60, np.inf],
                              labels=[1, 2, 3, 4, 5, 6, 7]).astype('int')
    df['MonthlyCharges_cat'] = pd.cut(df['MonthlyCharges'],
                                      bins=[15, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
                                      labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).astype('int')
    df['TotalCharges_cat'] = pd.cut(df['TotalCharges'],
                                    bins=[0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000],
                                    labels=[1, 2, 3, 4, 5, 6, 7, 8, 9]).astype('int')
    # Encode Label column with dummy variable (0,1)

    df['Churn_cat'] = (df['Churn'] == 'Yes').astype(int)

    df['SeniorCitizen'] = df['SeniorCitizen'].astype('object')

    # Create category bins for continuous data ('tenure', 'Monthly Charges', 'Total Charges')

    df['tenure_cat'] = pd.cut(df['tenure'], bins=[0, 10, 20, 30, 40, 50, 60, np.inf],
                              labels=[1, 2, 3, 4, 5, 6, 7]).astype('int')

    df['MonthlyCharges_cat'] = pd.cut(df['MonthlyCharges'],
                                      bins=[15, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
                                      labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).astype('int')

    df['TotalCharges_cat'] = pd.cut(df['TotalCharges'],
                                    bins=[0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000],
                                    labels=[1, 2, 3, 4, 5, 6, 7, 8, 9]).astype('int')

    return df


def show_tree(tree_classifier, features, path):
    f = io.StringIO()
    export_graphviz(tree_classifier, out_file=f, feature_names=features)
    pydotplus.graph_from_dot_data(f.getvalue()).write_png(path)
    img = imageio.imread(path)
    plt.rcParams["figure.figsize"] = (20, 20)
    plt.imshow(img)


def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.savefig('./Images/ROC_y_train_y_scores.png')
    # plt.show()


def plot_feature_importance(df, model):
    n_features = df.shape[1]
    plt.figure(figsize=(12, 8))
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), df.columns)
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)
    plt.savefig('./Images/Feature_importance.png')
    # plt.show()
