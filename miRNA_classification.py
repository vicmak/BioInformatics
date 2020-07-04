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

files = [filepath1, filepath2, filepath3, filepath4, filepath5]
#filepath = "/Users/macbook/Downloads/miRNA_byBMI_forR.txt"

def swap_columns(df, c1, c2):
    df['temp'] = df[c1]
    df[c1] = df[c2]
    df[c2] = df['temp']
    df.drop(columns=['temp'], inplace=True)


for file_path in files:

    data = pd.read_csv(file_path, sep='\t')
    array = data.values
    features_number = len(data.columns)
    Xs = array[:,1:features_number]
    ys = array[:,0]

    RF = RandomForestClassifier(n_estimators=1, max_depth=2, random_state=1 )
    RF = RandomForestClassifier()
    LR = LogisticRegression()
    LR = LogisticRegression(random_state=0, solver='liblinear',max_iter=1000, dual=True)
    SVM = svm.LinearSVC(max_iter=5000)
    SVM = svm.LinearSVC(random_state=1)
    NB = GaussianNB()
    MLP = MLPClassifier(hidden_layer_sizes=(100,50,25),activation='logistic')
    MLP = MLPClassifier(random_state=1)
    DT = DecisionTreeClassifier(random_state=1)
    GBC = GradientBoostingClassifier(random_state=1)

    num_instances = len(Xs)
    loocv = model_selection.LeaveOneOut()
    results = model_selection.cross_val_score(SVM, Xs, ys, cv=loocv)
    print (file_path)
    print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))

#RF = RandomForestClassifier(n_estimators=2, max_depth=2, random_state=0)


#Xs = data.iloc[:,1:1339]
#ys = data.iloc[:,0]


#print(ys)
#print(Xs)

#SVM = svm.LinearSVC()
#SVM.fit(Xs,ys)


#RF = RandomForestClassifier(n_estimators=2, max_depth=2, random_state=0)
#RF.fit(Xs, ys)




#RF.predict(Xs.iloc[3:14,:])

#print(round(RF.score(Xs,ys), 4))

#LR = LogisticRegression(random_state=0, solver='lbfgs',max_iter=190).fit(Xs, ys)
#print("Logistic regression")
#print(round(LR.score(Xs,ys), 4))
#data=data.set_index('Geneid')
#print(data)
#transposed_data = data.T

#print(transposed_data)
#transposed_data.insert(0, 'index' range(0, len(transposed_data)))
#transposed_data.set_index('index')
#print(transposed_data)
#print("sli ced")
#Xs = transposed_data.iloc[:,2:1337]
#print(transposed_data.iloc[1:,0])

#print(transposed_data.iloc[1:,1:1338])
#print(transposed_data.columns.values)


#Xs = transposed_data.iloc[:,[1,1338]]
#ys = transposed_data.loc['Geneid']

#print("Data")
#print(Xs)
#print(ys)

#kmeans = KMeans(n_clusters=2).fit(transposed_data)
#centroids = kmeans.cluster_centers_

# Nice Pythonic way to get the indices of the points for each corresponding cluster
#mydict = {i: np.where(kmeans.labels_ == i)[0] for i in range(kmeans.n_clusters)}

#print("in cluster 0:")
#print(mydict)
#print("Centroids")
#print(centroids)