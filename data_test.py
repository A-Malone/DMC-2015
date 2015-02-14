import numpy as np
import csv

file_name  = "../Data/SEM_DAILY_BUILD.csv"

dic = {}
counter = 0

def get_one_hot(kw_text):
    kws = txt.split("+")
    for kw in kws:
        if(kw == ""):
            continue
        kw_index = dic[int(kw[2:])]
        search_data[ind][kw_index] = 1


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
    for i in kws:
        if(i == ""):
            continue
        try:
            a = dic[int(i[2:])]
        except:
            dic[int(i[2:])] = kw_counter
            kw_counter += 1

#input_file.close()

print("Done pre-test: {} kws , {} clickthroughs , {} lines".format(kw_counter , click_line_count, line_count))


#Create data storage
search_data = np.zeros((click_line_count+1, kw_counter))
search_target = np.zeros(click_line_count+1)


#keys = sorted(dic, key=dic.get)
#for kw in keys:
#    print("{} : {}".format(kw, dic[kw]))

input_file = csv.DictReader(open(file_name, "r"))
useful_row_counter = 0
for ind, row in enumerate(input_file):
    if(int(row["CLICKS"]) == 0):        
        continue
    useful_row_counter += 1

    txt = row["KEYWD_TXT"]
    kws = txt.split("+")
    for kw in kws:
        if(kw == ""):
            continue
        kw_index = dic[int(kw[2:])]
        search_data[useful_row_counter][kw_index] = 1

    try:
        search_target[useful_row_counter] = int(row["APPLICATIONS"])/float(row["CLICKS"])
    except:
        search_target[useful_row_counter] = 0.0

print("Done loading data : {} {}".format(search_data.shape , search_target.shape))


#----SVM implementation

print("Running through support vector machine")

from sklearn import svm
from sklearn import datasets
#import matplotlib.pyplot as plt
#from sklearn.externals import joblib

import time

start = time.clock()

n = 200
clf = svm.SVC(gamma=0.001, C=100.)
#clf = svm.SVC()
a = clf.fit(search_data[:n], search_target[:n])

modelled = time.clock()
print(modelled-start)

b = clf.predict(search_data[n+2])
print(b)

print(time.clock() - modelled)

#print(np.array(a))
#np.loadtxt(open(file_name,"rb"),delimiter=",",skiprows=1)

