from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from helpers import plot_feature_importance, plot_partial_dependence, prepare_data
from collections import Counter
import matplotlib.pyplot as plt



path = r'data\dataset_telecom_01.csv'

X, y, df, y_df = prepare_data(path)

clf = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, subsample=0.5, max_depth=2, random_state=0)
scores = cross_val_score(clf, X, y, cv=5)
print(scores)  # check if model has learned any pattern


clf = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1,
                                    subsample=0.5, max_depth=2, random_state=0).fit(X, y)

plot_feature_importance(clf, df.columns, top_most_important=7) # show top features which had influence on churn


plot_partial_dependence(clf, X, df.columns, ['Factor_03'])  # Dependence of feature and churn
plot_partial_dependence(clf, X, df.columns, ['MonthlyCharges']) # Dependence of feature and churn


for col in df.columns:
    plot_partial_dependence(clf, X, df.columns, ['C_03_Electronic check'] + [col]) # Dependence of feature and churn



### Trying to find out how payment types distributed due to months of serive ###
for col_ in ['C_03_Bank transfer (automatic)', 'C_03_Credit card (automatic)', 'C_03_Electronic check', 'C_03_Mailed check']:
    for churned_ in [1]:
        temp_counter = Counter(df['Factor_03'][(df[col_] == 1)&(y_df == churned_)].values)
        total_occ = sum(temp_counter.values())
        points = [(k, v / total_occ) for k, v in temp_counter.items()]
        points.sort(key=lambda _:_[0])
        plt.plot([i[0] for i in points], [i[1] for i in points], "-",
                 label='{}_{}_{}'.format(col_, "churned", churned_), markersize=1, alpha=0.7)

plt.title('Distribution of Payment type by months')
plt.xlabel('Month')
plt.ylabel('Payments made in given month')
plt.legend()

