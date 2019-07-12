import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
plt.figure(0).clf()

accuracy_list = []
precision_list = []
recall_list = []
f1_score_list = []
cohens_kappa_list = []
ROC_AUC_list = []
true_negative = []
false_negative = []
false_positive = []
true_positive = []


for i in range(5):

    train=np.load("./SVM2/small/training_"+str(i+1)+".npy", allow_pickle=True)
    x_train=np.array([i[0] for i in train]).reshape(-1,1024)
    y_train=np.array([i[1] for i in train])
    print(x_train.shape, y_train.shape)

    y_train=y_train[:, 1]

    test=np.load("./SVM2/small/testing_"+str(i+1)+".npy", allow_pickle=True)
    x_test=np.array([i[0] for i in test]).reshape(-1,1024)
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

    fpr, tpr, thresh = roc_curve(y_test, y_prob[:, 1])
    plt.plot(fpr, tpr, label="fold "+str(i+1)+", auc="+str(auc), linestyle='dashed')

    true_negative.append(matrix[0][0])
    false_negative.append(matrix[1][0])
    false_positive.append(matrix[0][1])
    true_positive.append(matrix[1][1])
    precision_list.append(precision)
    recall_list.append(recall)
    f1_score_list.append(f1)
    cohens_kappa_list.append(kappa)
    ROC_AUC_list.append(auc)
    accuracy_list.append(accuracy * 100)



plt.legend(loc=0)
plt.savefig("./ROC_CURVES/SVM2_small")
plt.close()


print("%.2f%% (+/- %.2f%%)" % (np.mean(accuracy_list), np.std(accuracy_list)))
print("%.2f (+/- %.2f)" % (np.mean(precision_list), np.std(precision_list)))
print("%.2f (+/- %.2f)" % (np.mean(recall_list), np.std(recall_list)))
print("%.2f (+/- %.2f)" % (np.mean(f1_score_list), np.std(f1_score_list)))
print("%.2f (+/- %.2f)" % (np.mean(cohens_kappa_list), np.std(cohens_kappa_list)))
print("%.2f (+/- %.2f)" % (np.mean(ROC_AUC_list), np.std(ROC_AUC_list)))
print("%.2f (+/- %.2f)" % (np.mean(true_negative), np.std(true_negative)))
print("%.2f (+/- %.2f)" % (np.mean(false_negative), np.std(false_negative)))
print("%.2f (+/- %.2f)" % (np.mean(false_positive), np.std(false_positive)))
print("%.2f (+/- %.2f)" % (np.mean(true_positive), np.std(true_positive)))

