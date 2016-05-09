#!/usr/bin/python

import sys
import pickle
from pprint import pprint

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score,recall_score,precision_score

import pandas as pd
import numpy as np


def genrateClassifiers():

    classifiers = []

    clf1 = RandomForestClassifier()
    param1 = {
        'n_estimators' : [10,20,50],
        'max_features' : [5, 10 ,15]
    }

    classifiers.append((clf1, param1))

    clf2 = LogisticRegression(class_weight="auto", random_state=24)
    param2 = {
         'penalty' : ['l1','l2'],

         'C' : [ 1e3,1e6,1e9],
         'tol' : [ 1e-3,1e-6,1e-10]
    }

    classifiers.append( (clf2,param2))

    clf3 = svm.SVC()
    param3 = {
        'C' : [1,1e3,1e6,1e9],
        'kernel' : ['linear', 'rbf'],
        'gamma' : [1,10,1e3]
    }

    # classifiers.append((clf3,param3))
    return classifiers


def optimize_all_classifier(clf_list,features_train, labels_train):
    "here we optimize all the classifiers and only save the best ones. "

    best_clf_list = []

    cross_validator = StratifiedShuffleSplit(y=labels_train, random_state=0)

    for clf,param in clf_list:
        best = GridSearchCV(clf,param, scoring="f1", cv=cross_validator)
        best = best.fit(features_train, labels_train)
        best = best.best_estimator_
        print best
        best_clf_list.append(best)


    rpf_score = []
    for clf in best_clf_list:
        predictions = clf.predict(features_test)
        r = recall_score(labels_test,predictions)
        f = f1_score(labels_test,predictions)
        p = precision_score(labels_test,predictions)
        rpf_score.append((r,p,f))


    return rpf_score, best_clf_list



### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'deferral_payments', 'loan_advances', \
                 'bonus', 'restricted_stock_deferred', 'deferred_income', \
                 'expenses', 'exercised_stock_options', \
                 'other', 'long_term_incentive', 'restricted_stock', 'director_fees',\
                 'to_messages', 'from_poi_to_this_person', \
                 'from_messages', 'from_this_person_to_poi', \
                  'shared_receipt_with_poi'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


#replace missing values with median for the purposes of this presentation



### Task 2: Remove outliers
outliers = ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK']

for outlier in outliers:
    data_dict.pop(outlier,0)


### Task 3: Create new feature(s)
"""
Add new features to the data_dict
"""

for name_key in data_dict:
    try:
        total_messages = data_dict[name_key]['from_messages'] + data_dict[name_key]['to_messages']
        poi_messages =  data_dict[name_key]["from_poi_to_this_person"] + data_dict[name_key]["from_this_person_to_poi"] + \
                        data_dict[name_key]["shared_receipt_with_poi"]
        ratio = poi_messages / (total_messages * 1.0)
        data_dict[name_key]['poi_ratio'] = ratio
    except:
        data_dict[name_key]['poi_ratio'] = 'NaN'


features_list = features_list + ['poi_ratio']

#now clean the features from NaN values. 2 strategies are used
# 1. Remove the columns which have greater that 50% NaN values
# 2. Impute the remaining NaN values to median

#pre processing the data for the NaN values
dataframe = pd.DataFrame.from_records(list(data_dict.values()))
persons = pd.Series(list(data_dict.keys()))

dataframe.replace(to_replace='NaN', value=np.nan, inplace=True)
#get the number of NaN values in the column.
print dataframe.isnull().sum()

# print features.isnull().sum()


### Store to my_dataset for easy export below.
my_dataset = data_dict

from sklearn.cross_validation import StratifiedShuffleSplit
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Do Some preprocessing like Scaling of features to a range between 0 and 1
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
# from sklearn.linear_model import LogisticRegression
#clf = GaussianNB()
#clf = DecisionTreeClassifier()


### Find the best (most discriminative) features to use 
#from sklearn.feature_selection import RFE
#rfe = RFE(estimator=clf,n_features_to_select=9,step = 1)
#rfe.fit(features,labels)
#print rfe.ranking_
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
features = SelectKBest(f_classif,k = 18).fit_transform(features,labels)


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html



### Generating the testing and training set. Stratified shuffle split is used because the data is less.
from sklearn.cross_validation import  StratifiedShuffleSplit
shuffle = StratifiedShuffleSplit(labels,10, test_size=0.3,random_state=42)

features_train = []
features_test = []
labels_train = []
labels_test = []
for train_index, test_index in shuffle:

    for ii in train_index:
        features_train.append(features[ii])
        labels_train.append(labels[ii])
    for jj in test_index:
        features_test.append(features[jj])
        labels_test.append(labels[jj])




#generate all the classfiers to work upon the data
clf_list = genrateClassifiers()

# get the rpf scores and the optimized classifiers from all the params
rpf_score, best_clf = optimize_all_classifier(clf_list,features_train,labels_train)
print rpf_score
# sort the scores to get the winning algo
f1_scores = [i[2] for i in rpf_score]

best = [i[0] for i in sorted(enumerate(f1_scores),key=lambda x : x[0])]

# the winning algo will be last in the list
winner_idx = best[-1]

clf = best_clf[winner_idx]
#get the winning algo
#Scaling is added into the pipeline because the classifier exported will not scale the features.
clf= Pipeline( steps = [('scaler',scaler),("classifer", clf)]);

#print clf
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)