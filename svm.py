import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

train=np.load("./SVM1/3D/training_5.npy", allow_pickle=True)
x_train=np.array([i[0] for i in train]).reshape(-1,512)
y_train=np.array([i[1] for i in train])
print(x_train.shape, y_train.shape)

y_train=y_train[:, 1]

test=np.load("./SVM1/3D/testing_5.npy", allow_pickle=True)
x_test=np.array([i[0] for i in test]).reshape(-1,512)
y_test=np.array([i[1] for i in test])
print(x_test.shape, y_test.shape)

y_test=y_test[:, 1]

clf=svm.SVC(gamma='scale', probability=True, random_state=42)

clf.fit(x_train, y_train)

y_pred=clf.predict(x_test)
y_prob=clf.predict_proba(x_test)

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %f' % accuracy)

# precision tp / (tp + fp)
precision = precision_score(y_test, y_pred)
print('Precision: %f' % precision)

# recall: tp / (tp + fn)
recall = recall_score(y_test, y_pred)
print('Recall: %f' % recall)

# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_test, y_pred)
print('F1 score: %f' % f1)
 
# kappa
kappa = cohen_kappa_score(y_test, y_pred)
print('Cohens kappa: %f' % kappa)

# ROC AUC
auc = roc_auc_score(y_test, y_prob[:, 1])
print('ROC AUC: %f' % auc)

# confusion matrix
matrix = confusion_matrix(y_test, y_pred)
print(matrix)


