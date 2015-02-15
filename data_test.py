#------------------------------------------------------------------------------
#----TORONTONENSIS
#----Aidan Malone, Jannis Mei, Justin Chiao, Justin Leung
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
#----DESCRIPTION:
# A Support Vector Machine that classifies keyword searches by conversion rate,
# returning the estimated conversion rate of the keyword search.
#
# Parameters for the support vector machine are chose through an exhaustive 
# grid search machine learning algorithm.
#
# This is implemented in python using the Scikit-learn, an open-source machine 
# learning package. It requires numpy, and scipy, all of which can be acquired
# using the python pip utility.

#------------------------------------------------------------------------------
#----RESULTS:
# Tentative results suggest a correlation coefficient R^2 of ~0.89 of the
# conversion rate.
#------------------------------------------------------------------------------



#------------------------------------------------------------------------------
#------------------------------PARSE DATA--------------------------------------
#------------------------------------------------------------------------------

#----Basic imports
import numpy as np
import csv
import time
import math

#The file from which the data is to be loaded
file_name  = "../Data/SEM_DAILY_BUILD.csv"


dic = {}
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
    """Returns average conversion rate"""
    try:
        return int(row["APPLICATIONS"])/float(row["CLICKS"])
    except:
        return 0.0

def chi_squared(m, y, v):
    """My own R^2 check"""
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
            kw_counter += 1


print("Done pre-test: {} kws , {} clickthroughs , {} lines".format(kw_counter , click_line_count, line_count))


#------------------------------------------------------------------------------
#-------------------------------LOAD DATA--------------------------------------
#------------------------------------------------------------------------------
min_bucket = -2;
base = 1.7

def get_bucket(rate):
    """ The clustering system, which determines the categories into which each search term is sorted """  
    global base, min_bucket

    if(rate==0):
        return min_bucket

    bucket = int(math.floor(math.log(rate*100.0, base)))
    if(bucket < min_bucket):
        bucket = min_bucket

    return bucket

def get_bucket_names():
    global base, min_bucket
    names = []
    b = min_bucket
    while(base**(b-1)<100):
        names.append("{:.2f} to {:.2f}".format(base**(b-1), base**(b)))
        b+=1
    return names


#Create datasets
search_data = np.zeros((click_line_count+1, kw_counter+non_kw_dimensions))
search_data_u = np.zeros((click_line_count+1, non_kw_dimensions+1))
search_target_r = np.zeros(click_line_count+1)
search_target_c = np.zeros(click_line_count+1)

#Open data file for reading
input_file = csv.DictReader(open(file_name, "r"))
useful_row_counter = 0

#----Load in the actual data
for ind, row in enumerate(input_file):
    if(int(row["CLICKS"]) == 0):
        continue
    useful_row_counter += 1

    #---Load non-keyword-data from the row
    udata = get_non_kw_data(row)

    for da, udatum in enumerate(udata):
        search_data[useful_row_counter][kw_counter + da] = udatum
        search_data_u[useful_row_counter][da] = udatum

   
    #----Load keyword data
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

#----CHOOSE DATA SET
to_train = search_data_scaled
to_target = search_target_c

#------------------------------------------------------------------------------
#---------------------------MODEL IMPLEMENTATION-------------------------------
#------------------------------------------------------------------------------

#----SKlearn imports
print("Importing SKlearn")
from sklearn import svm, datasets, feature_selection, cross_validation, linear_model, neighbors
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA

#----Optional debugging imports
import matplotlib.pyplot as plt
#from sklearn.externals import joblib


#Set up a list of classifiers such that they can be run in succession
classifier_list = []

#---CLASSIFERS
svc_rbf = svm.SVC(C=1000, gamma=0.01, probability=True)          #Gauss SVC
#classifier_list.append(svc_rbf)                #Good one

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
classifier_list.append(Pipeline([('reduce_dim', PCA()), ('svm', svc_rbf)]))
#svc_lin = svm.SVC(kernel='linear')            #Linear SVC
#classifier_list.append(RFE(estimator=svc_lin, n_features_to_select=1, step=1))
#classifier_list.append(p1)



#----CONSTANTS
n = 8000            #Number of data points to train on
v = 3000            #number of data points to validate w/

k_folding = False   #Whether or not cross-validation will be run
folds = 6           #If so, run with this many data folds

def run_model_list(classifier_list, to_train, to_target, n, v, k_folding, folds):
    """ Trains all models in the classifier list with the data provided """    
    
    verbose = False      #Whether or not to print out all the data

    print("Training and testing models")

    for clf in classifier_list:

        #Which type of classifier is running
        print("Running {}\n".format(type(clf)))        
        
        if(k_folding):      #K-FOLDING
            print("Running K-Folding: {} folds on {} rows".format(folds, n))
            this_scores = cross_validation.cross_val_score(clf, to_train[:n], to_target[:n], n_jobs=-1, cv=folds)
            print("k_folding results: {}\n".format(this_scores))
        else:               #LINEAR TRAINING            
            start = time.clock()
            a = clf.fit(to_train[:n], to_target[:n])
            modelled = time.clock()
            print("Modelling time: {}s".format(modelled-start))

            print("Running data prediction")
            predicted_data = clf.predict(to_train[n+1: n+v+1])
            pred_av = np.mean([predicted_data[j] for j in range(len(predicted_data))], axis = 0)
            #plt.plot(pred_av)

            report = classification_report(to_target[n+1: n+v+1],predicted_data, target_names=get_bucket_names())

            if(verbose):
                for i in range(v):
                    plt.plot(predicted_data[i] - pred_av)
                    print("{:.3f} - {}".format(to_target[n+1+i], predicted_data[i]))
                plt.show()
        

            R2 = clf.score(to_train[n+1: n+v+1], to_target[n+1: n+v+1])
            print("Official R squared: {} -- {}\n".format(R2,type(clf)))
            print(report)
            print("Prediction time: {}s".format(time.clock() - modelled))


def grid_search(to_target, to_train):
    """ Exhaustive grid search to find optimal parameters for SVC """
    global n

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

#Runs when the file is run
if(__name__ == "__main__"):
    run_model_list(classifier_list, to_train, to_target, n, v, k_folding, folds)
    #grid_search(to_target, to_train)
