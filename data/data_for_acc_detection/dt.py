# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 18:35:22 2018

@author: User
"""


import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix  
import pickle



def get_code(tree, feature_names, target_names,
             spacer_base="    "):
    """Produce psuedo-code for decision tree.

    Args
    ----
    tree -- scikit-leant DescisionTree.
    feature_names -- list of feature names.
    target_names -- list of target (class) names.
    spacer_base -- used for spacing code (default: "    ").

    
    """
    left      = tree.tree_.children_left
    right     = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features  = [feature_names[i] for i in tree.tree_.feature]
    value = tree.tree_.value

    def recurse(left, right, threshold, features, node, depth):
        spacer = spacer_base * depth
        if (threshold[node] != -2):
            print(spacer + "if ( " + features[node] + " <= " + \
                  str(threshold[node]) + " ) {")
            if left[node] != -1:
                    recurse(left, right, threshold, features,
                            left[node], depth+1)
            print(spacer + "}\n" + spacer +"else {")
            if right[node] != -1:
                    recurse(left, right, threshold, features,
                            right[node], depth+1)
            print(spacer + "}")
        else:
            target = value[node]
            for i, v in zip(np.nonzero(target)[1],
                            target[np.nonzero(target)]):
                target_name = target_names[i]
                target_count = int(v)
                print(spacer + "return " + str(target_name) + \
                      " ( " + str(target_count) + " examples )")

    recurse(left, right, threshold, features, 0, 0)



df1 = pd.read_csv(r'acc_proc.csv')
df2 = pd.read_csv(r'acc_proc_less7.csv')


del df1['Date']
del df1['Unnamed: 0']
del df1['index']
del df2['Date']
del df2['Unnamed: 0']
del df2['index']
del df2['level_0']


df1['g_on'] = 1
df2['g_on'] = 0


df = pd.concat([df1, df2], ignore_index=True)

#df = shuffle(df).reset_index(drop=True)

use_cols = ['stdx', 'stdy', 'stdz', 'meanx', 'meany', 'meanz', 'medx', 'medy', 'medz' ]


features = list(df[use_cols].columns)
print(features)

y = df['g_on']
X = df[features]

#print(pd.DataFrame.corr(X))



#
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)

#pd.plotting.scatter_matrix(X_train, c=y_train, figsize=(15, 15), hist_kwds={'bins': 20}, s=60)
#

dt = DecisionTreeClassifier(min_samples_split=20)
dt.fit(X_train, y_train)
#
#with open("tree.txt", "w") as f:
#    f = export_graphviz(dt, out_file=f, feature_names=features)
#
y_pred = dt.predict(X_test)  

print(dt.score(X_test, y_test))
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))

#prob = (dt.predict_proba(X))
#for i in range(len(X)):
#    print("X=%s, Predicted=%s" % (list(X.iloc[i]), prob[i]))

#
pickle_path = 'acc_model2.pkl'
model_pkl = open(pickle_path, 'wb')
pickle.dump(dt, model_pkl)
model_pkl.close()

#targets = df['g_on'].unique()
#get_code(dt, features, targets)

