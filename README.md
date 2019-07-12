# Detection of Lymph Nodes in Mediastinum region

Thanks to Professor Yuji Iwahori and Hiroyasu Usami for their constant help and support.

This repository contains codes in Python3 developed by author for the abovementioned topic.

Our best model achieves a recall of 0.83 (+/- 0.05) at about 1-2 False Positives per CT Volume on an average with ROC of 0.93 (+/- 0.01) which is a drastic improvement on current State-of-the-Art with a recall of 0.70 at 3 False Positives per CT volume with ROC of 0.90. ROC curves are provided in the repository for quick reference. 

We also provide exhaustive analysis of CNN, 3D-CNN, CNN-SVM, 3D-CNN-SVM, CNN-TTA (Test Time Augmentation) and CNN-TTA-SVM on LN detection task.

Visuals on different Data Augmentation techniques are also provided.


Note :

a) To obtain weights of trained U-Net with test Dice Coefficient of 0.60 for LNs candidate generation as stated in algorithm.

b) To obtain CNN/3D-CNN features used for training SVM.

Please contact following mail : jayan170108018@iitg.ac.in
