import numpy as np
import csv

file_name  = "../Data/SEM_DAILY_BUILD.csv"

#USEFUL_VARS = ["TOTAL_QUALITY_SCORE"]

dic = {}
kw_cvr_dic = {}
counter = 0

def get_one_hot(kw_text):
    kws = txt.split("+")
    for kw in kws:
        if(kw == ""):
            continue
        kw_index = dic[int(kw[2:])]
        search_data[ind][kw_index] = 1

non_kw_dimensions = 3
def get_non_kw_data(d):
    a = np.array([0.0, 0.0, 0.0])
    try:
        a[2] = int(d["CONDITIONAL_IMPRESSIONS"])
        a[1] = float(d["IMPRESSION_TOTAL_RANK"])
        a[0] = int(d["TOTAL_QUALITY_SCORE"]) / float(d["IMPRESSIONS"])
    except:
        pass
    return a

def get_conversion_ratio(row):
    try:
        return int(row["APPLICATIONS"])/float(row["CLICKS"])        
    except:
        return 0.0

def chi_squared(m, y):
    #Make my own R2
    y_var = np.var(y)
    t = 0
    for i in range(v):
        t += (y[i] - m[i])**2
    return 1 - t / y_var / v


#----Read in Data first time
line_count = 0
click_line_count = 0
kw_counter = 0

input_file = csv.DictReader(open(file_name, "r"))
for row in input_file:    
    if(line_count == 0): #Skip first row
        line_count += 1
        continue

    #Count lines
    line_count += 1

    if(int(row["CLICKS"]) != 0):        
        click_line_count += 1
    else:
        continue            #Skip if no clicks
    
    #Count keywords
    txt = row["KEYWD_TXT"]
    kws = txt.split("+")
    
    cvr = get_conversion_ratio(row)

    for i in kws:
        if(i == ""):
            continue
        try:
            a = dic[int(i[2:])]            
        except:
            dic[int(i[2:])] = kw_counter
            kw_cvr_dic[int(i[2:])] = cvr
            kw_counter += 1

#input_file.close()

print("Done pre-test: {} kws , {} clickthroughs , {} lines".format(kw_counter , click_line_count, line_count))


#Create data storage
search_data = np.zeros((click_line_count+1, kw_counter+non_kw_dimensions))
search_target = np.zeros(click_line_count+1)

search_data_u = np.zeros((click_line_count+1, non_kw_dimensions+1))


#keys = sorted(dic, key=dic.get)
#for kw in keys:
#    print("{} : {}".format(kw, dic[kw]))

input_file = csv.DictReader(open(file_name, "r"))
useful_row_counter = 0

#----Load in the actual data
for ind, row in enumerate(input_file):
    if(int(row["CLICKS"]) == 0):
        continue
    useful_row_counter += 1

    #---Load the data from the row
    udata = get_non_kw_data(row)
    for da, udatum in enumerate(udata):
        search_data[useful_row_counter][kw_counter + da] = udatum
        search_data_u[useful_row_counter][da] = udatum     

   
    #Keyword_data
    txt = row["KEYWD_TXT"]
    kws = txt.split("+")
    for kw in kws:
        if(kw == ""):
            continue
        kw_index = dic[int(kw[2:])]
        search_data_u[-1] += kw_cvr_dic[int(i[2:])]

        search_data[useful_row_counter][kw_index] = 1
    
    search_target[useful_row_counter] = get_conversion_ratio(row)
    

print("Done loading data : {} {}".format(search_data.shape , search_target.shape))

#----DATA NORMALIZATION
from sklearn import preprocessing
search_data_scaled =  preprocessing.scale(search_data)
search_data_u_scaled =  preprocessing.scale(search_data_u)
search_target_scaled =  preprocessing.scale(search_target)

#----CHOSEN_DATA_SET
to_use = 0


#----SVM IMPLEMETATION


print("Importing models")
from sklearn import svm, datasets, feature_selection, cross_validation, linear_model
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier

print("Training and testing model")
#import matplotlib.pyplot as plt
#from sklearn.externals import joblib

classifier_list = []
classifier_list.append(svm.SVR(C=1e2, gamma=0.01))
#classifier_list.append(svm.SVC(C=1e2, gamma=0.1))
#classifier_list.append(linear_model.Lasso(alpha = 0.1))
classifier_list.append(DecisionTreeRegressor(max_depth=5))
#classifier_list.append(AdaBoostClassifier(n_estimators=100))
#classifier_list.append(GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0))
#classifier_list.append(linear_model.LassoLars(alpha=.1))
#classifier_list.append(RandomForestClassifier(n_estimators=10))

#clf = Pipeline([
#  ('feature_selection', LinearSVC(penalty="l1", dual=False)),
#  ('classification', DecisionTreeRegressor(max_depth=3))
#])

import time

start = time.clock()


n = 2000    #Number of data points to train on
v = 200     #number of data points to validate w/

for clf in classifier_list:

    a = clf.fit(search_data_u[:n], search_target[:n])

    #Cross-reference
    #this_scores = cross_validation.cross_val_score(clf, search_data_u_scaled[:n], search_target[:n], n_jobs=-1, cv=5)

    modelled = time.clock()
    print("Modelling time: {}s".format(modelled-start))

    #print(this_scores)

    #print("{} {} {}".format(search_data_u[n+1], search_target[n+1] ,clf.predict(search_data_u[n+1])))

    searched = clf.predict(search_data_u[n+1: n+v+1])


    R2 = clf.score(search_data_u[n+1: n+v+1], search_target[n+1: n+v+1])
    print("Prediction time: {}s".format(time.clock() - modelled))
    print("R squared: {} -- {}\n".format(R2,type(clf)))






#for ind, pred in enumerate(prediction):
#    print("P: {} , A: {}".format(pred, search_target[n+1+ind]))

