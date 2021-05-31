import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
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
    sencit = {0: 'No',
              1: 'Yes'}
    df = df.replace({"SeniorCitizen": sencit})

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


# Training Set Size on Error Stability

def plot_learning_curves(model, X, y, num_samples):
    #X_train, X_test, X_valid, y_train, y_test, y_valid = train_test_valid(X, y)
    X_train, X_test, y_train, y_test, = train_test(X, y)
    train_errors, test_errors = [], []
    #valid_errors = []
    for m in range(1, int(num_samples)):
        model.fit(X_train[:m], y_train[:m])
        y_predict = model.predict(X_train[:m])
        y_test_predict = model.predict(X_test)
        #y_valid_predict = model.predict(X_valid)
        train_errors.append(mean_squared_error(y_train[:m], y_predict))
        test_errors.append(mean_squared_error(y_test, y_test_predict))
        #val_errors.append(mean_squared_error(y_valid, y_valid_predict))

    plt.plot(np.sqrt(train_errors), "r-+", lw=1, label="train")
    plt.plot(np.sqrt(test_errors), "b--", lw=2, label="test")
    #plt.plot(np.sqrt(val_errors), "g-", lw=2, label='validation')


def train_test_valid(X, y):
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_trainval, y_trainval, test_size=0.2,  random_state=42)
    return X_train, X_test, X_valid, y_train, y_test, y_valid


def train_test(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42) #
    return X_train, X_test, y_train, y_test