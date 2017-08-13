import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import partial_dependence as pdep
import pandas as pd


def plot_feature_importance(tmodel, col_names, top_most_important = 10):

    importance_rate = tmodel.feature_importances_
    importance_tuple = [(n, v) for n, v in zip(col_names, importance_rate)]

    top_tuple = sorted(importance_tuple, key=lambda _: _[1], reverse=True)[:top_most_important]

    fig, ax = plt.subplots()

    ax.barh(range(top_most_important), [_[1] for _ in top_tuple], align='center', color='green', ecolor='black')
    ax.set_yticks(range(top_most_important))
    ax.set_yticklabels([_[0] for _ in top_tuple])
    ax.invert_yaxis()
    ax.set_xlabel('Importance')
    ax.set_title('Feature importance')
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.show()


def plot_partial_dependence(tmodel, X, col_names, cols_to_plot):
    assert isinstance(cols_to_plot, list)
    assert len(cols_to_plot) < 3

    inds = [np.where(col_names == col)[0][0] for col in cols_to_plot]

    if len(inds) == 2:
        features = (inds[0], inds[1], (inds[0], inds[1]))
        fig, axs = pdep.plot_partial_dependence(tmodel, X, features, feature_names=col_names)
    else:
        features = [inds[0]]
        fig, axs = pdep.plot_partial_dependence(tmodel, X, features, feature_names=col_names)


def prepare_data(path):
    data = pd.read_csv(path)

    # for col in data.columns:
    #     print(col, data[col].unique())

    bin_col_list = ['Factor_01', 'Factor_02', 'Service_01', 'C_02', 'is_churn']
    cat_col_list = ['Service_02', 'Service_03', 'Service_04', 'Service_05', 'service_06', 'Service_07', 'Service_08',
                    'Service_09', 'C_01', 'C_03']
    dis_col_list = ['Factor_03', 'MonthlyCharges', 'Charges']
    bin_col_list1 = ['Factor_00']

    data['gend'] = (data['gend'] == 'Male').astype(int)  # is_male

    for col in bin_col_list:
        data[col] = (data[col] == 'Yes').astype(int)

    # data[dis_col_list] = (data[dis_col_list] - data[dis_col_list].mean()) / data[dis_col_list].std()

    df = pd.concat([data[['gend'] + bin_col_list1 + bin_col_list + dis_col_list], pd.get_dummies(data[cat_col_list])],
                   axis=1)

    df['tel_services'] = df[['Service_01', 'Service_02_Yes']].sum(axis=1)
    df['internet_services'] = df[['Service_03_DSL', 'Service_03_Fiber optic', 'Service_04_Yes', 'Service_05_Yes',
                                  'service_06_Yes', 'Service_07_Yes', 'Service_08_Yes', 'Service_09_Yes']].sum(axis=1)
    df['less_30'] = (df['MonthlyCharges'] < 30).astype(int)
    df['less_60'] = ((df['MonthlyCharges'] >= 30) & (df['MonthlyCharges'] < 60)).astype(int)
    df['over_60'] = (df['MonthlyCharges'] >= 60).astype(int)

    y_df = df.pop('is_churn')
    df.pop('Charges')

    X = df.values
    np.nan_to_num(X, copy=False)  # Charges NaN to 0

    y = y_df.values

    return X, y, df, y_df


def graph_overfitting(model, X_test, y_test, params):

    test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

    for i, y_pred in enumerate(model.staged_predict(X_test)):
        test_score[i] = model.loss_(y_test, y_pred)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Deviance')
    plt.plot(np.arange(params['n_estimators']) + 1, model.train_score_, 'b-',
             label='Training Set Deviance')
    plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
             label='Test Set Deviance')
    plt.legend(loc='upper right')
    plt.xlabel('Boosting Iterations')
    plt.ylabel('Deviance')