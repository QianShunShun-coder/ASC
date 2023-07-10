import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold, RepeatedKFold, GridSearchCV
from scipy.stats import pearsonr, ttest_ind,levene
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
data_1 = pd.read_excel(r"/Users/chu/Desktop/dataAQC.xlsx")
data_2 = pd.read_excel(r"/Users/chu/Desktop/dataSCC.xlsx")
data_3 = pd.read_excel(r"/Users/chu/Desktop/dataSSC.xlsx")
data_4 = pd.read_excel(r"/Users/chu/Desktop/dataADC.xlsx")
rows_1,__ = data_1.shape
rows_2,__ = data_2.shape
rows_3,__ = data_3.shape
rows_4,__ = data_4.shape
data_1.insert(0,'diagnosis',[0]*rows_1)
data_2.insert(0,'diagnosis',[1]*rows_2)
data_3.insert(0,'diagnosis',[2]*rows_3)
data_4.insert(0,'diagnosis',[3]*rows_4)
data = pd.concat([data_1,data_2,data_3,data_4])
data = shuffle(data)
data = data.fillna(0)
X = data[data.columns[1:]]
y = data['diagnosis']
colNames = X.columns
X = X.astype(np.float64)
X = StandardScaler().fit_transform(X)
X = pd.DataFrame(X)
X.columns = colNames
counts = 0
index = []
alphas = np.logspace(-3,1,50)
lassocv = LassoCV(alphas=alphas, cv=10, max_iter=100000).fit(X,y)
print(lassocv.alpha_)
coef = pd.Series(lassocv.coef_, index=X.columns)
print('Lasso picked' + str(sum(coef !=0)) + 'Variables and eliminated the other' + str(sum(coef == 0)))
index = coef[coef != 0].index
X = X[index]
print(coef[coef != 0])
Cs = np.logspace(-1,3,10,base=2)
gammas = np.logspace(-4,1,50,base=2)
param_grid = dict(C = Cs, gamma = gammas)
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
grid = GridSearchCV(svm.SVC(kernel='rbf'),param_grid=param_grid,cv=10).fit(x_train,y_train)
print(grid.best_params_)
C = grid.best_params_['C']
gamma = grid.best_params_['gamma']
print(C)
print(gamma)
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
model_svm = svm.SVC(kernel='rbf', C=C, gamma=gamma,probability=True).fit(x_train, y_train)
score_svm = model_svm.score(x_test,y_test)
print(score_svm)
rkf = RepeatedKFold(n_splits=5, n_repeats=2)
for train_index, test_index in rkf.split(X):
    x_train = X.iloc[train_index]
    x_test = X.iloc[test_index]
    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]
    model_svm = svm.SVC(kernel='rbf', C=C, gamma=gamma, probability=True).fit(x_train, y_train)
    score_svm = model_svm.score(x_test, y_test)
    print(score_svm)
model_rf = RandomForestClassifier(n_estimators=20).fit(x_train, y_train)
score_rf = model_rf.score(x_test, y_test)
print(score_rf)