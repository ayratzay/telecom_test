from helpers import prepare_data, graph_overfitting
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

path = r'data\dataset_telecom_01.csv'

X, y, df, _ = prepare_data(path)
months = df.pop("Factor_03")
df['MonthlyCharges'] = np.log(df['MonthlyCharges'] + 1)
X = df.values
y_months = months.values

### Predicting after how many months a client will churn ###
X_train, X_test, y_train, y_test = train_test_split(X, y_months, test_size=0.4, random_state=0)

params = {'n_estimators': 500, 'max_depth': 2, 'learning_rate': 0.01, 'subsample': 0.5, 'loss': 'ls'}
reg_model = GradientBoostingRegressor(**params)

reg_model.fit(X_train, y_train)
mае_model = mean_absolute_error(y_test, clf.predict(X_test))

base_model_results = np.ones(y_test.shape) * y_train.mean() # Base_model is just an average all churns
mае_basemodel = mean_absolute_error(y_test, base_model_results)

print("MAE Model: %.4f, MAE Base: %.4f" % (mае_model, mае_basemodel)) # Checking if new model is better then base model

preds_ = [(i1, i2, i3) for (i1, i2, i3) in zip(y_test, reg_model.predict(X_test), base_model_results)]
preds = sorted(preds_, key=lambda _: _[0])


plt.plot([i[0] for i in preds], label='True', alpha=0.7)
plt.plot([i[1] for i in preds], label='Pred', alpha=0.7) # Graphing periods to chech where model might be a weak predictor
plt.plot([i[2] for i in preds], label='Base', alpha=0.7)


graph_overfitting(reg_model, X_test, y_test, params)


### Final model ####
final_model = GradientBoostingRegressor(**params)
final_model.fit(X, y_months)

