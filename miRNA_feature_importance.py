import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn import model_selection

import numpy as np
from sklearn.cluster import KMeans

filepath1 = "/Users/macbook/Desktop/corpora/miRNA_Datasets/Obesity.txt"
filepath2 = "/Users/macbook/Desktop/corpora/miRNA_Datasets/GSE75473.txt"
filepath3 = "/Users/macbook/Desktop/corpora/miRNA_Datasets/GSE112804.txt"
filepath4 = "/Users/macbook/Desktop/corpora/miRNA_Datasets/GSE126386.txt"
filepath5 = "/Users/macbook/Desktop/corpora/miRNA_Datasets/GSE129373.txt"

#files = [filepath1, filepath2, filepath3, filepath4, filepath5]
files = [filepath5]

#filepath = "/Users/macbook/Downloads/miRNA_byBMI_forR.txt"

def swap_columns(df, c1, c2):
    df['temp'] = df[c1]
    df[c1] = df[c2]
    df[c2] = df['temp']
    df.drop(columns=['temp'], inplace=True)

def dt_feature_importance(model,normalize=True):

    left_c = model.tree_.children_left
    right_c = model.tree_.children_right

    impurity = model.tree_.impurity
    node_samples = model.tree_.weighted_n_node_samples

    # Initialize the feature importance, those not used remain zero
    feature_importance = np.zeros((model.tree_.n_features,))

    for idx,node in enumerate(model.tree_.feature):
        if node >= 0:
            # Accumulate the feature importance over all the nodes where it's used
            feature_importance[node]+=impurity[idx]*node_samples[idx]- \
                                   impurity[left_c[idx]]*node_samples[left_c[idx]]-\
                                   impurity[right_c[idx]]*node_samples[right_c[idx]]

    # Number of samples at the root node
    feature_importance/=node_samples[0]

    if normalize:
        normalizer = feature_importance.sum()
        if normalizer > 0:
            feature_importance/=normalizer

    return feature_importance

for file_path in files:

    data = pd.read_csv(file_path, sep='\t')
    array = data.values
    features_number = len(data.columns)
    Xs = array[:,1:features_number]
    ys = array[:,0]

    RF = RandomForestClassifier(n_estimators=10, max_depth=2, random_state=1 )
    rf = RandomForestClassifier()
    #RF = RandomForestClassifier()
    LR = LogisticRegression(C=0.1)
    #LR = LogisticRegression(random_state=0, solver='liblinear',max_iter=1000, dual=True)
    SVM = svm.LinearSVC(max_iter=5000)
    SVM = svm.LinearSVC(random_state=1)
    NB = GaussianNB()
    MLP = MLPClassifier(hidden_layer_sizes=(100,50,25),activation='logistic')
    MLP = MLPClassifier(random_state=1)
    DT = DecisionTreeClassifier(random_state=1)
    GBC = GradientBoostingClassifier(random_state=1)

    print("Decision tree feature importance:")
    print(file_path)
    DT.fit(Xs,ys)
    RF.fit(Xs,ys)
    GBC.fit(Xs,ys)
    features = DT.tree_.feature[DT.tree_.feature >= 0]  # Feature number should not be negative, indicates a leaf node
    #print(sorted(zip(features, dt_feature_importance(DT, False)[features]), key=lambda x: x[1], reverse=True))

    DT_importances = DT.feature_importances_
    RF_importances = RF.feature_importances_
    GBC_importances = GBC.feature_importances_

    DT_important_list = sorted(enumerate(DT_importances), key=lambda x: x[1], reverse=True)
    RF_important_list = sorted(enumerate(RF_importances), key=lambda x: x[1], reverse=True)
    GBC_important_list = sorted(enumerate(GBC_importances), key=lambda x: x[1], reverse=True)

    print("DT important features:")
    print(DT_important_list)
    print("RF important features:")
    print(RF_important_list)
    print("GBC important features:")
    print(GBC_important_list)

    #for i, v in enumerate(DT_importances):
        #if v!=0:
            #print('Feature: %0d, Score: %.5f' % (i, v))