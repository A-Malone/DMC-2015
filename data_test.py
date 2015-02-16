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

#----Basic imports
import numpy as np
import csv
import time
import math
#import matplotlib.pyplot as plt

#----SKlearn imports
print("Importing SKlearn..")
from sklearn import svm, datasets, feature_selection, cross_validation, linear_model, neighbors
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA

from sklearn import preprocessing
from sklearn.externals import joblib

#------------------------------------------------------------------------------
#----------------------------------MODEL---------------------------------------
#------------------------------------------------------------------------------
class CVR_Model(object):
    """ The model that produces models for the convergence """

    #----Classifiers
    classifier_list = []

    #----Datasets
    search_data = None         #Complete training data    
    search_data_scaled = None
    
    search_data_u = None       #Continuous training data
    search_data_u_scaled = None

    search_target_r = None     #Regression target data
    search_target_c = None     #Classification target data

    def __init__(self):
        super(CVR_Model, self).__init__()        

    #------------------------------------------------------------------------------
    #----------------------------MODEL BUILDING------------------------------------
    #------------------------------------------------------------------------------    

    def build_model(self, file_name):
        self.init_esitmators()      #Setup the estimators

        #Load in data
        self.load_data(file_name)

        #----CONSTANTS
        n = 1000               #Number of data points to train on
        v = 3000               #number of data points to validate w/

        self.train_model(self.search_data_scaled[:n], self.search_target_c[:n])

        self.classification_report(self.search_data_scaled[n+1:n+v+1], self.search_target_c[n+1:n+v+1], False)

        #k_folding = False       #Whether or not cross-validation will be run
        #folds = 6               #If so, run with this many data folds

    #----MODEL TRAINING
    #------------------------------------------------------------------------------    
    def train_model(self, to_train, to_target):
        """ Trains all models in the classifier list with the data provided """    

        print("Training and testing models")

        for clf in self.classifier_list:
            #Which type of classifier is running
            start = time.clock()
            print("Running {}".format(type(clf)))
            clf.fit(to_train,to_target)
            print("Trained {} in {}s\n".format(type(clf), time.clock() - start))
            
    
    #----MODEL EVALUATION
    #------------------------------------------------------------------------------    
    def k_fold_evaluate(self, X, y, folds):
        print("Running K-Folding: {} folds on {} rows".format(folds, len(X)))
        this_scores = cross_validation.cross_val_score(clf, X, y, n_jobs=-1, cv=folds)
        print("k_folding results: {}\n".format(this_scores))

    def classification_report(self, X, y, verbose):
        start = time.clock()
        
        for clf in self.classifier_list:
            print("Running data prediction for {}".format(type(clf)))

            predicted_data = clf.predict(X)
            #Useful for analysis
            #pred_av = np.mean([predicted_data[j] for j in range(len(predicted_data))], axis = 0)        

            report = classification_report(y, predicted_data, target_names=self.get_bucket_names())

            if(verbose):
                for i in range(v):
                    #plt.plot(predicted_data[i] - pred_av)
                    print("{:.3f} - {}".format(y[i], predicted_data[i]))
                #plt.show()
        

            R2 = clf.score(X, y)
            print("Official R squared: {} -- {}\n".format(R2,type(clf)))
            print(report)
            print("Prediction time: {}s".format(time.clock() - start))

    #----MODEL OPTIMIZATION
    #------------------------------------------------------------------------------ 
    def grid_search(self, to_target, to_train):
        """ Exhaustive grid search to find optimal parameters for SVC """        

        # Set the parameters by cross-validation
        tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                             'C': [1, 10, 100, 1000]},
                            {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

        scores = ['precision', 'recall']

        for score in scores:
            print("# Tuning hyper-parameters for %s \n" % score)           

            clf = GridSearchCV(svm.SVC(C=1), tuned_parameters, cv=5, scoring=score)
            clf.fit(to_train, to_target)

            print("Best parameters set found on development set:\n")
            print(clf.best_estimator_)            
            print("Grid scores on development set:\n")
            
            for params, mean_score, scores in clf.grid_scores_:
                print("%0.3f (+/-%0.03f) for %r"
                      % (mean_score, scores.std() / 2, params))
            print("\n")


    #----ESTIMATOR SETUP
    #------------------------------------------------------------------------------
    def init_esitmators(self):
        """ Our approach allows for comparison of multiple models on one data sets """

        #----The primary SKlearn estimator
        svc_rbf = svm.SVC(C=1000, gamma=0.01, probability=True)          #Gauss SVC
        self.classifier_list.append(Pipeline([('reduce_dim', PCA()), ('svm', svc_rbf)]))

        #----Inactive, tested, estimators

        #----CLASSIFERS        
        #classifier_list.append(svc_rbf)
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

        #----PIPELINES        
        #svc_lin = svm.SVC(kernel='linear')            #Linear SVC
        #classifier_list.append(RFE(estimator=svc_lin, n_features_to_select=1, step=1))        


    #----DATA LOADING
    #------------------------------------------------------------------------------
    kw_dict = {}    
    non_kw_dimensions = 8

    def load_data(self, file_name):
        line_count = 0
        click_line_count = 0
        kw_counter = 0
        counter = 0

        input_file = csv.DictReader(open(file_name, "r"))
        for row in input_file:
            
            line_count += 1                     #Count lines

            if(int(row["CLICKS"]) != 0):         #Skip if no clicks
                click_line_count += 1
            else:
                continue           
            
            #Count keywords
            txt = row["KEYWD_TXT"]
            kws = txt.split("+")
            
            for i in kws:
                if(i == ""):
                    continue
                try:
                    a = self.kw_dict[int(i[2:])]            
                except:
                    self.kw_dict[int(i[2:])] = kw_counter            
                    kw_counter += 1

        print("Done pre-test: {} kws , {} clickthroughs , {} lines".format(kw_counter , click_line_count, line_count))

        #Create datasets
        self.search_data = np.zeros((click_line_count+1, kw_counter+self.non_kw_dimensions))
        self.search_data_u = np.zeros((click_line_count+1, self.non_kw_dimensions+1))
        self.search_target_r = np.zeros(click_line_count+1)
        self.search_target_c = np.zeros(click_line_count+1)

        #Reopen data file for reading
        input_file = csv.DictReader(open(file_name, "r"))
        useful_row_counter = 0

        #----Load in the actual data
        for ind, row in enumerate(input_file):
            if(int(row["CLICKS"]) == 0):
                continue
            useful_row_counter += 1

            #---Load non-keyword-data from the row
            udata = self.get_non_kw_data(row)

            for da, udatum in enumerate(udata):
                self.search_data[useful_row_counter][kw_counter + da] = udatum
                self.search_data_u[useful_row_counter][da] = udatum

           
            #----Load keyword data
            txt = row["KEYWD_TXT"]
            kws = txt.split("+")
            for kw in kws:
                if(kw == ""):
                    continue
                kw_index = self.kw_dict[int(kw[2:])]

                self.search_data[useful_row_counter][kw_index] = 1
            
            self.search_target_c[useful_row_counter] = self.get_bucket(self.get_conversion_ratio(row))
            self.search_target_r[useful_row_counter] = self.get_conversion_ratio(row)
            

        print("Done loading data : {} {}".format(self.search_data.shape , self.search_target_r.shape))

        #----DATA NORMALIZATION        
        self.search_data_scaled =  preprocessing.scale(self.search_data)
        self.search_data_u_scaled =  preprocessing.scale(self.search_data_u)        

    #----BUCKETING
    #------------------------------------------------------------------------------
    min_bucket = -2
    base = 1.7
    def get_bucket(self, rate):
        """ The clustering system, which determines the categories into which each search term is sorted """          

        if(rate==0):
            return self.min_bucket

        bucket = int(math.floor(math.log(rate*100.0, self.base)))
        if(bucket < self.min_bucket):
            bucket = self.min_bucket

        return bucket

    def get_bucket_names(self):
        names = []
        b = self.min_bucket
        while(self.base**(b-1)<100):
            names.append("{:.2f} to {:.2f}".format(self.base**(b-1), self.base**(b)))
            b+=1
        return names

    #----DATA EXTRACTION
    #------------------------------------------------------------------------------
    def get_non_kw_data(self, d):
        a = np.zeros(self.non_kw_dimensions)
        
        a[7] = float(d["VISITS"]) if d["VISITS"] != "" else 0.0


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

    #------------------------------------------------------------------------------
    def get_conversion_ratio(self, row):
        """Returns average conversion rate"""
        try:
            return int(row["APPLICATIONS"])/float(row["CLICKS"])
        except:
            return 0.0
    
    #------------------------------------------------------------------------------
    #---------------------------MODEL APPLICATION-----------------------------------
    #------------------------------------------------------------------------------
    def run_model(self, data_file):
        pass

    #------------------------------------------------------------------------------
    #-------------------------------MODEL I/O--------------------------------------
    #------------------------------------------------------------------------------
    def save_model(self, outfiles):        
        for i,n in enumerate(infiles):
            joblib.dump(self.classifier_list[i], n)

    def load_model(self, infiles):
        for n in infiles:
            classifier_list.append(joblib.load(n))


#The file from which the data is to be loaded
file_name  = "../Data/SEM_DAILY_BUILD.csv"

def chi_squared(m, y, v):
    """My own R^2 check"""
    y_var = np.var(y)
    t = 0
    for i in range(v):
        t += (y[i] - m[i])**2
    return 1 - t / y_var / v


#Runs when the file is run
if(__name__ == "__main__"):
    model = CVR_Model()
    model.build_model(file_name)