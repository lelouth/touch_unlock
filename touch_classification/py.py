import time
from sklearn import metrics
import pickle as pickle
import pandas as pd
import numpy as np


# Multinomial Naive Bayes Classifier
def naive_bayes_classifier(train_x, train_y):
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB(alpha=0.01)
    model.fit(train_x, train_y)
    return model


# KNN Classifier
def knn_classifier(train_x, train_y):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier()
    model.fit(train_x, train_y)
    return model


# Logistic Regression Classifier
def logistic_regression_classifier(train_x, train_y):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty='l2')
    model.fit(train_x, train_y)
    return model


# Random Forest Classifier
def random_forest_classifier(train_x, train_y):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=8)
    model.fit(train_x, train_y)
    return model


# Decision Tree Classifier
def decision_tree_classifier(train_x, train_y):
    from sklearn import tree
    model = tree.DecisionTreeClassifier()
    model.fit(train_x, train_y)
    return model

# SVM Classifier
def svm_classifier(train_x, train_y):
    from sklearn.svm import SVC
    model = SVC(kernel='linear', probability=True)
    model.fit(train_x, train_y)
    return model

# def one_svm_classifier(train_x, train_y):
#     from sklearn.svm import OneClassSVM
#     model = OneClassSVM(kernel='rbf')
#     model.fit(train_x, train_y)
#     return model


# SVM Classifier using cross validation
def svm_cross_validation(train_x, train_y):
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
    grid_search = GridSearchCV(model, param_grid, n_jobs=1, verbose=1)
    grid_search.fit(train_x, train_y)
    best_parameters = grid_search.best_estimator_.get_params()
    # for para, val in list(best_parameters.items()):
    #     print(para, val)
    model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)
    model.fit(train_x, train_y)
    return model

def LR_cross_validation(train_x, train_y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV

    model = LogisticRegression(penalty='l2')
    param_grid = {'penalty': ['l1','l2'], 'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]}
    z = GridSearchCV(model, param_grid, n_jobs=1, verbose=1)
    z.fit(train_x, train_y)
    best_parameters = z.best_estimator_.get_params()
    model = LogisticRegression(C=best_parameters['C'], penalty=best_parameters['penalty'])

    model.fit(train_x, train_y)
    return model



def read_data(data_file):
    data = pd.read_csv(data_file, header=None, sep=' ')
    test = data.sample(frac=0.3, replace= True)
    z = pd.concat([test, data])
    train = z.drop_duplicates(keep=False)

    train_y = train.iloc[:,63]
    train_x = train.drop(63 , axis=1)
    test_y = test.iloc[:,63]
    test_x = test.drop(63 , axis=1)

    return train_x, train_y, test_x, test_y

#5,18,21,34,37,50,53
if __name__ == '__main__':
    datafilename = 'all.csv'
    data_file = "/home/lelouth/Desktop/S3/ML/project/" + datafilename
    data = pd.read_csv(data_file, header=None, sep=' ')

    thresh = 0.5
    model_save_file = 1
    model_save = {}
    test_classifiers = ['NB', 'KNN', 'LR', 'RF','DT', 'SVM', 'S2VMCV', 'LRCV']
    classifiers = {
                    'NB': naive_bayes_classifier,
                    'KNN': knn_classifier,
                    'LR': logistic_regression_classifier,
                    'RF': random_forest_classifier,
                    'DT': decision_tree_classifier,
                    'SVM': svm_classifier,
                   'SVMCV': svm_cross_validation,
                   'LRCV': LR_cross_validation,
                   }
    train_x, train_y, test_x, test_y = read_data(data_file)

    for classifier in test_classifiers:
        print('******************* %s ********************' % classifier)
        start_time = time.time()
        model = classifiers[classifier](train_x, train_y)
        print('training took %fs!' % (time.time() - start_time))
        predict = model.predict(test_x)
        if model_save_file != None:
            model_save[classifier] = model
        precision = metrics.precision_score(test_y, predict)
        recall = metrics.recall_score(test_y, predict)
        print('precision: %.2f%%, recall: %.2f%%' % (100 * precision, 100 * recall))
        F1 = metrics.f1_score(test_y, predict)
        print('F-1: %.2f%%' % (100 * F1))


    if model_save_file != None:
        pickle.dump(model_save, open(model_save_file, 'wb'))