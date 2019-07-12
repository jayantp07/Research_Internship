import numpy as np

accuracy_list=[0.821076, 0.831452, 0.840252, 0.785101, 0.770646]
precision_list=[0.908092, 0.819699, 0.876676, 0.757100, 0.843983]
recall_list=[0.747533, 0.829392, 0.801471, 0.807345, 0.706250]
f1_score_list=[0.820027, 0.824517, 0.837388, 0.781416, 0.768998]
cohens_kappa_list=[0.645436, 0.662387, 0.681022, 0.570508, 0.544810]
ROC_AUC_list=[0.911796, 0.904670, 0.924523, 0.877909, 0.853934]
true_negative=[922, 1080, 682, 1308, 1036]
false_positive=[92, 216, 92, 402, 188]
false_negative=[307, 202, 162, 299, 423]
true_positive=[909, 982, 654, 1253, 1017]

print("%.2f (+/- %.2f)" % (np.mean(accuracy_list), np.std(accuracy_list)))
print("%.2f (+/- %.2f)" % (np.mean(precision_list), np.std(precision_list)))
print("%.2f (+/- %.2f)" % (np.mean(recall_list), np.std(recall_list)))
print("%.2f (+/- %.2f)" % (np.mean(f1_score_list), np.std(f1_score_list)))
print("%.2f (+/- %.2f)" % (np.mean(cohens_kappa_list), np.std(cohens_kappa_list)))
print("%.2f (+/- %.2f)" % (np.mean(ROC_AUC_list), np.std(ROC_AUC_list)))
print("%.2f (+/- %.2f)" % (np.mean(true_negative), np.std(true_negative)))
print("%.2f (+/- %.2f)" % (np.mean(false_negative), np.std(false_negative)))
print("%.2f (+/- %.2f)" % (np.mean(false_positive), np.std(false_positive)))
print("%.2f (+/- %.2f)" % (np.mean(true_positive), np.std(true_positive)))
