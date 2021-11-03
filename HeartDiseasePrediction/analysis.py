from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.metrics import classification_report

unclean_df = pd.read_csv(r'C:\Users\User\Desktop\Applied Research Project\HeartDiseasePrediction (8) (1)\HeartDiseasePrediction\HeartDiseasePrediction\HeartDiseasePrediction\heart.csv')

unclean_df = unclean_df.drop_duplicates(subset=None, keep='first', inplace=False, ignore_index=False)
unclean_df['age'] = unclean_df['age'].apply(lambda x: round(x))

unclean_df['age'] = unclean_df['age'].apply(lambda x: round(x))

unclean_df.to_csv('cleaned_data2.csv')
df = unclean_df
original_data = df
X = df.drop(columns=['target'])
y = df['target']
SelectKBest_features = SelectKBest(mutual_info_classif, ).fit(X, y)

SelectKBest_features.get_support(indices=True)
X_method3 = X.iloc[:, SelectKBest_features.get_support(indices=True)]
X_method3

X_train3, X_test3, y_train3, y_test3 = train_test_split(X_method3, y, test_size=0.3, random_state=42)
logreg3 = LogisticRegression(max_iter=5000)
logreg3.fit(X_train3, y_train3)
y_pred_3 = logreg3.predict(X_test3)
print('Accuracy of logistic regression classifier on test set: {:.6f}'.format(logreg3.score(X_test3, y_test3)))
new_features = X_method3
new_features
ss = StandardScaler()

zscore = ss.fit_transform(new_features)

X_scaled = pd.DataFrame(zscore, index=new_features.index, columns=new_features.columns)
X_scaled = X_scaled.reset_index(drop=True)
X_scaled

X_train1, X_test1, y_train1, y_test1 = train_test_split(X_scaled, y, test_size=0.25, random_state=0)

X_train = X_train1
y_train = y_train1
X_test = X_test1
y_test = y_test1
AdaBoost_Classifier = AdaBoostClassifier(random_state=0).fit(X_train1, y_train1)
ada_scores = cross_val_score(AdaBoost_Classifier, X_train, y_train, cv=3)
ada_scores_max = ada_scores.max()
ada_scores_max

AdaBoost_Classifier_pred = AdaBoost_Classifier.predict(X_test1)

AdaBoost_Classifier_acc = accuracy_score(y_test1, AdaBoost_Classifier_pred)
AdaBoost_Classifier_acc
AdaBoost_Classifier_matrix = confusion_matrix(y_test1, AdaBoost_Classifier_pred)
AdaBoost_Classifier_matrix

print(classification_report(y_test1, AdaBoost_Classifier_pred))
