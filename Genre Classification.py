# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 15:18:08 2019

@author: SEVVAL
"""
#                   *** CLASSİFİCATİON ***

import pandas as pd
data1 = pd.read_csv("data.csv")
data = data1.drop(['filename'],axis=1)
#%%  korelasyon
corr=data.corr()

#%% split test-train
y=data.label
x=data.drop('label',axis=1)
#%% string olan label değerlerini sayısal değere dönüştürdü
from sklearn.preprocessing import LabelEncoder
genre_list = data.iloc[:, -1]
encoder = LabelEncoder()
y = encoder.fit_transform(genre_list)
#%% normalizasyon
import numpy as np
from sklearn.preprocessing import  StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(np.array(x, dtype = float))
x = (x - np.min(x))/(np.max(x)-np.min(x))
#%% datasetimizin train ve test olarak ayılması ½80(800 sample) train %20 test (200 sample)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=0)


#%%
dat = (X-np.min(X))/(np.max(X)-np.min(X))
#%% Visualizing test and train test
import numpy as np
import matplotlib.pyplot as plt
values, count = np.unique(np.argmax(y_train), return_counts=True)
plt.bar(values, count)

values, count = np.unique(np.argmax(y_test), return_counts=True)
plt.bar(values, count)
plt.show()

#%% KNN algoritmasindaki K degerinin belirlenmesi
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
grid = {"n_neighbors":np.arange(1,50)}
knn= KNeighborsClassifier()

knn_cv = GridSearchCV(knn, grid, cv = 10)  
knn_cv.fit(x,y)


print("tuned hyperparameter K: ",knn_cv.best_params_)
print("tuned parametreye gore en iyi accuracy (best score): ",knn_cv.best_score_)
#%%
# search for an optimal value of k for KNN
from sklearn.model_selection import cross_val_score
k_range = range(1, 31)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, x, y, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
#%%
# 10-fold (cv=10) cross-validation with K=6 (n_neighbors=6) for KNN 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
knn = KNeighborsClassifier(n_neighbors=6)
scores = cross_val_score(knn, x, y, cv=10, scoring='accuracy')
print(scores)
print(scores.mean())
#%%        KNN
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=6,weights='distance',algorithm='brute',metric='euclidean')  # k = n_neighbors

knn.fit(x_train,y_train)
prediction = knn.predict(x_test)
y_pred = knn.predict(x_test)
from sklearn import metrics
from sklearn.metrics import confusion_matrix

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print( confusion_matrix(y_test,y_pred))
from sklearn.metrics import classification_report
print("knn:", classification_report(y_test,y_pred))


#%%     RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# RandomForestClassifier sınıfından bir nesne ürettik
# n_estimators = Oluşturulacak karar ağacı sayısıdır. Değiştirildiğinde başarı oranıda değişti, en yüksek accuracy i   veren estimators değeri secildi.
rfc = RandomForestClassifier(n_estimators=24,random_state=0,max_features='sqrt')
rfc.fit(x_train,y_train)
rf=rfc.feature_importances_

# Test veri kümemizi verdik ve tahmin işlemini gerçekleştirdik
result2 = rfc.predict(x_test)
accuracy1 = accuracy_score(y_test, result2)
print(accuracy1)
print("Random Forest:")
print(classification_report(y_test,y_pred))

#%%  SVM parameter tuning for C
from sklearn.svm import SVC
result_kernel= []
a=('linear','sigmoid','poly','rbf')
for kernel in a:
    clf = SVC(gamma='auto',kernel=kernel,C=14)
    clf.fit(x_train,y_train) 
    result_kernel.append(clf.score(x_test,y_test))
    
 # en iyi sonuç linear de veriyor

#%% SVM parameter tuning for C
import matplotlib.pyplot as plt
C_range=list(range(1,26))

acc_score=[]
for c in C_range:
    svc = SVC(kernel='linear', C=c)
    scores = cross_val_score(svc, x, y, cv=10, scoring='accuracy')
    acc_score.append(scores.mean())
print(acc_score)  
C_values=list(range(1,26))
# plot the value of C for SVM (x-axis) versus the cross-validated accuracy (y-axis)
plt.plot(C_values,acc_score)
plt.xticks(np.arange(0,27,2))
plt.xlabel('Value of C for SVC')
plt.ylabel('Cross-Validated Accuracy')
#%%   SVC
svc1 = SVC(kernel='linear', C=14)
svc1.fit(x_train,y_train) 

predictionsvc = svc1.predict(x_test)
y_predsvc = svc1.predict(x_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_predsvc))
print( confusion_matrix(y_test,y_predsvc))
print("SVM:", classification_report(y_test,y_predsvc))

# Model Accuracy, how often is the classifier correct?
#%%  Naive Bayes
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train,y_train) 

from sklearn.naive_bayes import MultinomialNB
clf1 = MultinomialNB()
clf1.fit(x_train,y_train)

from sklearn.naive_bayes import ComplementNB
clf2 = ComplementNB()
clf2.fit(x_train,y_train)

print("\n","GaussianNB:",nb.score(x_test,y_test),"\n","MultinomialNB:",clf1.score(x_test,y_test),"\n","ComplementNB:",clf2.score(x_test,y_test))
# en uygunu accuracy i yüksek olduğu için GaussianNB seçildi
predictionnb = nb.predict(x_test)
y_prednb = nb.predict(x_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_prednb))
print( confusion_matrix(y_test,y_prednb))
print("GaussianNB")
print(classification_report(y_test,y_prednb))
#%%  Decision Tree
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion="entropy", max_depth=None,min_samples_split=10,max_features=18,random_state=0)
dt = dt.fit(x_train,y_train)
predictiondt = dt.predict(x_test)
y_preddt = dt.predict(x_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_preddt))
print( confusion_matrix(y_test,y_preddt))
print("Decision Tree")
print(classification_report(y_test,y_preddt))

#%%  Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
#logreg = LogisticRegression()
#%%
from sklearn import model_selection
models = []
#models.append(('LR', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier(n_neighbors=6,weights='distance',algorithm='brute',metric='euclidean')))
models.append(('CART', DecisionTreeClassifier(criterion="entropy", max_depth=None,min_samples_split=10,max_features=18,random_state=0)))
models.append(('RF', RandomForestClassifier(n_estimators=24,random_state=0,max_features='sqrt')))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(kernel='linear', C=14)))
results = []
names = []
scoring = 'accuracy'
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=0)
	cv_results = model_selection.cross_val_score(model, x_train, y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f" % (name, cv_results.mean())
	print(msg)
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

#%%
