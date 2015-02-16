Capital One Data Mining Cup 2015 - TORONTONENSIS
======
Aidan Malone, Jannis Mei, Justin Chiao, Justin Leung


Description:
------------
A Support Vector Machine that classifies keyword searches by conversion rate,
returning the estimated conversion rate of the keyword search.
Parameters for the support vector machine are chose through an exhaustive 
grid search machine learning algorithm.
This is implemented in python using the Scikit-learn, an open-source machine 
learning package. It requires numpy, and scipy, all of which can be acquired
using the python pip utility.


Results:
----------
Tentative results suggest a correlation coefficient R^2 of ~0.89 of the
conversion rate.


Instructions:
------------
1. Install dependancies: pip install numpy scipy scikit-learn
⋅⋅*Note: matplotlib can also be used in debugging, but is not necessary
2. Run this file using: python data_test.py
⋅⋅*Note: Do not change the relative locations of any files in the directory, they are all reference relatively.
3. Results will be written to data_out.csv, in the same folder as this file



Todo/Wishlist:
------------
1. Weighting inputs based on number of impressions
2. Better parameter search
3. Try different dimensionality reduction algorithms (Kernel_PCA)

