import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, precision_score, \
    recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

path = 'path_to_directory'

kNN_clf = KNeighborsClassifier()
svc_clf = SVC(gamma='auto')

classifiers = [clone(kNN_clf),
               clone(svc_clf)]

for i in range(10):
    X_train = np.load('{}/{}/pca_train_images.npy'.format(path, i))
    X_test = np.load('{}/{}/pca_test_images.npy'.format(path, i))
    y_train = np.load('{}/{}/pca_train_labels.npy'.format(path, i))
    y_test = np.load('{}/{}/pca_test_labels.npy'.format(path, i))

    for classifier in classifiers:
        classifier.fit(X_train, y_train.squeeze())
        prediction = classifier.predict(X_test)
        acc_score = round(accuracy_score(y_test, prediction), 5)
        fscore = round(f1_score(y_test, prediction), 5)
        precision = precision_score(y_test, prediction)
        recall = recall_score(y_test, prediction)
        if isinstance(classifier, KNeighborsClassifier):
            print("Fold {}.\nAccuracy of kNN classifier is {} ".format(i,
                                                                       acc_score))
            print("Fscore of kNN classifier is {} ".format(fscore))
            print("Precision: {}".format(precision))
            print("Recall: {}".format(recall))
        elif isinstance(classifier, SVC):
            print("Accuracy of SVC classifier is {} ".format(acc_score))
            print("Fscore of SVC classifier is {} ".format(fscore))
            print("Precision: {}".format(precision))
            print("Recall: {}".format(recall))
