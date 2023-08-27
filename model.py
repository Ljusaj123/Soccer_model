import pandas as pd
import numpy as np
import time

import pickle
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


df = pd.read_csv('./data/matches.csv')

#dropping columns one wouldn't have before an actual match

cols_to_drop = ['season', 'match_name','date', 'home_team', 'away_team', 'home_score', 'away_score',
                'h_match_points', 'a_match_points' ]

df.drop( columns = cols_to_drop, inplace = True)

# df = df.drop(df[df.winner == 'DRAW'].index)


#filling NAs
df.fillna(-33, inplace = True)

#turning the target variable into integers
df['winner'] = np.where(df.winner == 'HOME_TEAM', 2, np.where(df.winner == 'AWAY_TEAM', 1, 0))


#turning categorical into dummy vars
df_dum = pd.get_dummies(df)


np.random.seed(101)

X = df_dum.drop(columns = ['winner'], axis = 1)
y = df_dum.winner.values

#splitting into train and test set to check which model is the best one to work on
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

#scaling features
scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

#creating models variable to iterate through each model and print result
models = [LogisticRegression(max_iter= 1000, multi_class = 'multinomial'),
          RandomForestClassifier(max_depth=30, n_estimators=30, max_features=3), 
          KNeighborsClassifier(n_neighbors = 20), 
          SVC(kernel="linear", C=0.02, probability=True)]

names = ['Logistic Regression', 'Random Forest', 'KNN', 'SVC']

#loop through each model and print train score and elapsed time
for model, name in zip(models, names):
    start = time.time()
    scores = cross_val_score(model, X_train, y_train ,scoring= 'accuracy', cv=5)
    print(name, ":", "%0.3f, +- %0.3f" % (scores.mean(), scores.std()), " - Elapsed time: ", time.time() - start)


#Creating loop to test which set of features is the best one for Logistic Regression

acc_results = []
n_features = []

#best classifier on training data
clf = LogisticRegression(max_iter= 1000, multi_class = 'multinomial') ##probaj za sve

for i in range(5 ,40):
    rfe = RFE(estimator = clf, n_features_to_select = i, step=5)
    rfe.fit(X, y)
    X_temp = rfe.transform(X)

    np.random.seed(101)

    X_train, X_test, y_train, y_test = train_test_split(X_temp,y, test_size = 0.2)

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    start = time.time()
    scores = cross_val_score(clf, X_train, y_train ,scoring= 'accuracy', cv=5)
    print(" Clf result :", "%0.3f, +- %0.3f" % (scores.mean(), scores.std()), 'N_features :', i)
    acc_results.append(scores.mean())
    n_features.append(i)

plt.plot(n_features, acc_results)
plt.ylabel('Accuracy')
plt.xlabel('N features')
plt.show()


#getting the best 13 features from RFE
rfe = RFE(estimator = clf, n_features_to_select = 15, step=1)
rfe.fit(X, y)
X_transformed = rfe.transform(X)

np.random.seed(101)
X_train, X_test, y_train, y_test = train_test_split(X_transformed,y, test_size = 0.2)

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


#getting column names
featured_columns = pd.DataFrame(rfe.support_,
                            index = X.columns,
                            columns=['is_in'])

featured_columns = featured_columns[featured_columns.is_in == True].index.tolist()


#tuning logistic regression
parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
 'fit_intercept': (True, False), 'solver' : ('newton-cg', 'sag', 'saga', 'lbfgs'), 'class_weight' : (None, 'balanced')}

gs = GridSearchCV(clf, parameters, scoring='accuracy', cv=3)
start = time.time()

#printing best fits and time elapsed
gs.fit(X_train,y_train)
print(gs.best_score_, gs.best_params_,  time.time() - start)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

svc = SVC(kernel="linear", C=0.02, probability=True)
svc.fit(X_train, y_train)



#testing models on unseen data 
tpred_lr = gs.best_estimator_.predict(X_test)
tpred_rf = rf.predict(X_test)
tpred_knn = knn.predict(X_test)
svc_pred = svc.predict(X_test)

test_models = [tpred_lr, tpred_rf, tpred_knn, svc_pred]

print(classification_report(y_test, tpred_lr, digits = 3))
print(classification_report(y_test, tpred_rf, digits = 3))
print(classification_report(y_test, tpred_knn, digits = 3))
print(classification_report(y_test, svc_pred, digits = 3))



#function to get winning odd value in simulation dataset
def get_winning_odd(df):
    if df.winner == 2:
        result = df.h_odd
    elif df.winner == 0:
        result = df.a_odd
    else:
        result = df.d_odd
    return result

#creating dataframe with test data to simulate betting winnings with models

test_df = pd.DataFrame(scaler.inverse_transform(X_test),columns =  featured_columns)
test_df['tpred_lr'] = tpred_lr
test_df['tpred_rf'] = tpred_rf
test_df['tpred_knn'] = tpred_knn
test_df['svc_pred'] = svc_pred

test_df['winner'] = y_test
test_df['winning_odd'] = test_df.apply(lambda x: get_winning_odd(x), axis = 1)

test_df['lr_profit'] = (test_df.winner == test_df.tpred_lr) * test_df.winning_odd * 100
test_df['rf_profit'] = (test_df.winner == test_df.tpred_rf) * test_df.winning_odd * 100
test_df['knn_profit'] = (test_df.winner == test_df.tpred_knn) * test_df.winning_odd * 100
test_df['svc_profit'] = (test_df.winner == test_df.svc_pred) * test_df.winning_odd * 100

investment = len(test_df) * 100

lr_return = test_df.lr_profit.sum() - investment
rf_return = test_df.rf_profit.sum() - investment
knn_return = test_df.knn_profit.sum() - investment
svc_return = test_df.svc_profit.sum() - investment

profit = (lr_return/investment * 100).round(2)
profit_rf = (rf_return/investment * 100).round(2)
profit_knn = (knn_return/investment * 100).round(2)
profit_svc = (svc_return/investment * 100).round(2)

print(f'''Logistic Regression return: ${lr_return}

Random Forest return: ${rf_return}

KNN return:  ${knn_return} \n

SVC return : ${svc_return}

Logistic Regression model profit percentage : {profit} %

Random Forest profit percentage: ${profit_rf}%


%KNN model profit percentage : {profit_knn} %

SVC profit percentage: ${profit_svc}%
''')


#retraining final model on full data
gs.best_estimator_.fit(X_transformed, y)

#Saving model and features
model_data = pd.Series( {
    'model': gs,
    'features': featured_columns
} )

pickle.dump(model_data, open("chatbot_model2.h5", 'wb'))


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

for model, name in zip(test_models, names):

    cm = confusion_matrix(y_test, model, labels= [0,1,2])
    cmd_obj = ConfusionMatrixDisplay(cm)
    cmd_obj.plot()
    cmd_obj.ax_.set(
                title=f'Matrica konfuzije {name}',
                xlabel='Predviđene oznake',
                ylabel='Točne oznake',  
                )
    plt.show()

