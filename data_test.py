import numpy as np
import csv

file_name  = "../Data/SEM_DAILY_BUILD.csv"

#USEFUL_VARS = ["TOTAL_QUALITY_SCORE"]

dic = {}
kw_cvr_dic = {}
counter = 0

non_kw_dimensions = 7
def get_non_kw_data(d):
    a = np.zeros(non_kw_dimensions)
    
    a[6] = 0 if d["DVIC_ID"]=="D" else 1
    a[5] = 0 if d["LANG_ID"]=="E" else 1
    a[5] = 0 if d["ENGN_ID"]=="G" else 1
    a[4] = 0 if d["MTCH_TYPE_ID"]=="B" else 1
    a[3] = float(d["IMPRESSIONS"])
    a[2] = int(d["CONDITIONAL_IMPRESSIONS"])

    try:
        a[1] = float(d["IMPRESSION_TOTAL_RANK"]) / float(d["IMPRESSIONS"])
        a[0] = int(d["TOTAL_QUALITY_SCORE"]) / float(d["IMPRESSIONS"])
    except:
        pass
    return a

def get_conversion_ratio(row):
    try:
        return int(row["APPLICATIONS"])/float(row["CLICKS"])
    except:
        return 0.0

def chi_squared(m, y, v):
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


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#----DATA GATHERING
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def get_bucket(rate):
    buckets = 20
    return int(round(rate*buckets))


#Create data storage
search_data = np.zeros((click_line_count+1, kw_counter+non_kw_dimensions))
search_data_u = np.zeros((click_line_count+1, non_kw_dimensions+1))
search_target_r = np.zeros(click_line_count+1)
search_target_c = np.zeros(click_line_count+1)


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
        #search_data_u[-1] += kw_cvr_dic[int(i[2:])]

        search_data[useful_row_counter][kw_index] = 1
    
    search_target_c[useful_row_counter] = get_bucket(get_conversion_ratio(row))
    search_target_r[useful_row_counter] = get_conversion_ratio(row)
    

print("Done loading data : {} {}".format(search_data.shape , search_target_r.shape))

#----DATA NORMALIZATION
from sklearn import preprocessing
search_data_scaled =  preprocessing.scale(search_data)
search_data_u_scaled =  preprocessing.scale(search_data_u)

#----CHOSEN_DATA_SET
to_train = search_data_scaled
to_target = search_target_c


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#----MODEL IMPLEMETATION
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


print("Importing SKlearn")
from sklearn import svm, datasets, feature_selection, cross_validation, linear_model, neighbors
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report

#import matplotlib.pyplot as plt
#from sklearn.externals import joblib

#Set up a list of classifiers such that they can be run in succession
classifier_list = []

#---CLASSIFERS
svc_rbf = svm.SVC(C=2e3, gamma=0.1)            #Gauss SVC
classifier_list.append(svc_rbf)                #Good one

#Add multiple classifers
#for c in range(5):
#    for y in range(-3,1):
#        classifier_list.append(svm.SVC(C=10**c, gamma=10**y))


#classifier_list.append(neighbors.KNeighborsClassifier(10))
#classifier_list.append(AdaBoostClassifier(n_estimators=100))
#classifier_list.append(GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0))
#classifier_list.append(RandomForestClassifier(n_estimators=10))

#----REGRESSORS
#classifier_list.append(linear_model.LassoLars(alpha=.1))
#classifier_list.append(svm.SVR(C=1e2, gamma=0.1))
#classifier_list.append(linear_model.Lasso(alpha = 0.1))
#classifier_list.append(DecisionTreeRegressor(max_depth=3))
#classifier_list.append(linear_model.SGDRegressor(learning_rate='constant', eta0=0.1))

#----COMBINATIONS
svc_lin = svm.SVC(kernel='linear')            #Gauss SVC
p1 = Pipeline([
  ('feature_selection', LinearSVC(penalty="l2", dual=False)),
  ('classification', svc_lin)
])
#classifier_list.append(RFE(estimator=svc_lin, n_features_to_select=1, step=1))
#classifier_list.append(p1)


#------------------------------------------------------------------------------
#----CONSTANTS

n = 1000     #Number of data points to train on
v = 6000     #number of data points to validate w/
k_folding = False
folds = 6

def run_model_list(to_train, to_target, classifier_list, n, v, k_folding, folds):  
    
    o = 0
    verbose = True


    import time

    print("Training and testing models")

    for clf in classifier_list:

        print("Running {}\n".format(type(clf)))    

        #----VALIDATION
        #Cross-reference
        if(k_folding):
            print("Running K-Folding: {} folds on {} rows".format(folds, n))
            this_scores = cross_validation.cross_val_score(clf, to_train[o:n+o], to_target[o:n+o], n_jobs=-1, cv=folds)
            print("k_folding results: {}\n".format(this_scores))
        else:
            #----TRAINING
            start = time.clock()
            a = clf.fit(to_train[o:n+o], to_target[o:n+o])
            modelled = time.clock()
            print("Modelling time: {}s".format(modelled-start))

            print("Running data prediction")
            predicted_data = clf.predict(to_train[n+1+o: n+v+1+o])
            #filtered_data = np.array([x if x>0.11 else 0.04 for x in predicted_data.tolist()])
            if(verbose):
                for i in range(v):
                    print("{:.3f} - {:.3f}".format(to_target[n+1+i+o], predicted_data[i]))   
        

            R2 = clf.score(to_train[n+1+o: n+v+1+o], to_target[n+1+o: n+v+1+o])        
            print("Original R squared: {} -- {}\n".format(R2,type(clf)))
            print("\nPrediction time: {}s".format(time.clock() - modelled))


def grid_search(to_target, to_train):
    # Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                         'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    scores = ['precision', 'recall']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(svm.SVC(C=1), tuned_parameters, cv=5, scoring=score)
        clf.fit(to_train[:n], to_target[:n])

        print("Best parameters set found on development set:")
        print()
        print(clf.best_estimator_)
        print()
        print("Grid scores on development set:")
        print()
        for params, mean_score, scores in clf.grid_scores_:
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean_score, scores.std() / 2, params))
        print()

if(__name__ == "__main__"):
