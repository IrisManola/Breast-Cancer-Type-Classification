#!/usr/bin/python3
import pandas as pd
import numpy as np
from scipy.io.arff import loadarff 
import matplotlib.pyplot as plt

# Load the data and explore
raw_data = loadarff('BreastCancerAll.reduced.using.cfs.missing.arff')
df = pd.DataFrame(raw_data[0])
#dfr.describe()
df.info()
# there are 59 variables in the dataset and 157 observations

#Do the data need cleaning?
#are there any missing values?
df.isnull().sum()
#this shows the list  of all variables and tells if there are missing values. 
#There are two variables that have 1 missing value each. Those we will drop from our data:
print ("Size and dimentions of dataset before cleaning: ", df.size, df.shape)
#Size of dataset before cleaning:  9420
df[['2170_chrom4_reg833085-843614_probnorm','2630_chrom4_reg187735989-188875290_probnorm']] = df[['2170_chrom4_reg833085-843614_probnorm','2630_chrom4_reg187735989-188875290_probnorm']].apply(pd.to_numeric, errors='coerce')
df = df.dropna()
df = df.reset_index(drop=True)
print ("Size and dimentions of dataset after cleaning: ", df.size, df.shape)
#Size of dataset after cleaning:  9300

# The cancer labels are categorical data. They need to be transformed in integers with a label encoder so that pythons models can understand them
from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
df['class'] = labelencoder_Y.fit_transform(df['class'])
Y = df['class']
X = df.iloc[:, 0:58]

# Let's see which are the 5 most informative features in the dataset, as those that explain most of the variance:
from sklearn.feature_selection import SelectKBest, f_classif
#Suppose, we select 5 features with top 5 Fisher scores
selector = SelectKBest(f_classif, k = 5)
#New dataframe with the selected features for later use in the classifier. fit() method works too, if you want only the feature names and their corresponding scores
X_new = selector.fit_transform(X, Y)
names = X.columns.values[selector.get_support()]
scores = selector.scores_[selector.get_support()]
names_scores = list(zip(names, scores))
ns_df = pd.DataFrame(data = names_scores, columns=['Feat_names', 'F_Scores'])
#Sort the dataframe for better visualization
ns_df_sorted = ns_df.sort_values(['F_Scores', 'Feat_names'], ascending = [False, True])
print(ns_df_sorted)

# the names of the 5 most informative features are printed, together with their relevant 
# F-scores (the ratio between the explained and the unexplained variance).


# how many of each cancer class is in our data:
print (df['class'].value_counts())
#gives:
#b'HER2+' 0 :  49
#b'HR+'   1 :  53
#b'TN'    2 :  53
#Name: class, dtype: int64
# so the 3 cancer types are well distributed withing our data

# Divide the data into the (assumed) independent variables X and the dependent Y (Y is named already a few lines above)
X = df.iloc[:, 0:58].values
Y = Y.values
# We will test now some fitting in this case supervised machine learning classification methods to see which one performs best here 

#First split the data into a train set and a test set in order at the end to test both the train and the test accuracy.
# Will be 80% train data 20% test

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)

#We need to bring all features to the same level of magnitudes. 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#import the tool that will test the accuracy of the models
from sklearn.metrics import classification_report, confusion_matrix


#k Nearest Neighbor algorithm
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, Y_train)
#predict the test set results and check the accuracy with each of our model:
Y_pred = classifier.predict(X_test)
#test the accuracy with a confusion matrix
from sklearn.metrics import confusion_matrix
cmknn = confusion_matrix(Y_test, Y_pred)
print ('Classification report kNN')
print (classification_report(Y_test, Y_pred))

#Naïve Bayes Algorithm

from sklearn.naive_bayes import GaussianNB
classifierNB = GaussianNB()
classifierNB.fit(X_train, Y_train)
Y_pred = classifierNB.predict(X_test)
cmnb = confusion_matrix(Y_test, Y_pred)
print ('Classification report Naive Bayes')
print (classification_report(Y_test, Y_pred))

#Decision Tree Algorithm

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)
cmdt = confusion_matrix(Y_test, Y_pred)
print ('Classification report Decision Tree') 
print (classification_report(Y_test, Y_pred))

#Random Forest Classification algorithm

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)
cmrf = confusion_matrix(Y_test, Y_pred)
print ('Classification report Random Forest')
print (classification_report(Y_test, Y_pred))

# a visual inspection of the confusion matrixes gives relatively similar accuracy between the different methods
# in the precision and recall (false positive and negatives) -non normalized- that follows in the code below. In the code above,
#the harmonic average F1 of the precision and recall (normalized) will gives the overall estimation of the accuracy. 

#Make a function that plots the confusion matrix
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# plot the classification reports!
plt.figure()
plot_confusion_matrix(cmknn, classes=['HR+','TN','HER2+'],normalize= False,  title='Confusion matrix - k Nearest Neighbor')
plt.savefig('cm_kNN.png')
plt.figure()
plot_confusion_matrix(cmnb, classes=['HR+','TN','HER2+'],normalize= False,  title='Confusion matrix - Naïve Bayes')
plt.savefig('cm_nb.png')
plt.figure()
plot_confusion_matrix(cmrf, classes=['HR+','TN','HER2+'],normalize= False,  title='Confusion matrix - Random Forest')
plt.savefig('cm_rf.png')
plt.figure()
plot_confusion_matrix(cmdt, classes=['HR+','TN','HER2+'],normalize= False,  title='Confusion matrix - Decision Tree')
plt.savefig('cm_dt.png')
#plt.show()

# And the winner is the Naive Bayes!!
 
