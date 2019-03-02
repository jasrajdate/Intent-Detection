#!/usr/bin/env python
# coding: utf-8

# In[338]:



import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from scipy.sparse import csr_matrix

get_ipython().run_line_magic('matplotlib', 'inline')


# In[339]:


# Data conversion
def getdata(file):
    rawdata = open(file,"r")

    messages = []
    intent = []
    for line in rawdata.readlines(): 
        sentence = line.split("\t")

        actual_words = sentence[0].split(" ")
        encoded_words = sentence[1].split(" ")

        for index, word in enumerate(encoded_words):
            if word == "O":
                encoded_words[index] = actual_words[index]

        msg = " ".join(encoded_words[1:-1])
        label = encoded_words[-1][0:-1]

        messages.append(msg)
        intent.append(label)

    data = pd.DataFrame(data={'message':messages,'intent':intent})
    return data


# In[340]:


train = getdata("atis-2.train.w-intent.iob.txt")
test = getdata("atis.test.w-intent.iob.txt")


# In[341]:


test.head()


# In[342]:


train.groupby('intent')['message'].nunique()


# In[343]:



## Clean Data
stops = set(stopwords.words("english"))
def cleandata(text, lowercase = False, remove_stops = False, stemming = False,lemmatize=False):
    txt = str(text)
   
    txt = re.sub(r'\n',r' ',txt)
    
    if lowercase:
        txt = " ".join([w.lower() for w in txt.split()])
        
    if remove_stops:
        txt = " ".join([w for w in txt.split() if w not in stops])
    
    if stemming:
        st = PorterStemmer()
        txt = " ".join([st.stem(w) for w in txt.split()])

    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        txt = " ".join([lemmatizer.lemmatize(w, pos='v') for w in txt.split()])
        
    return txt


# In[344]:



## Clean data 
trainClean = train['message'].map(lambda x: cleandata(x, lowercase=True,remove_stops=True, stemming=True, lemmatize=True))
testClean = test['message'].map(lambda x: cleandata(x, lowercase=True,remove_stops=True, stemming=True, lemmatize =True))

# Feature extraction
vectorizer = TfidfVectorizer(analyzer='word', min_df=0.0, max_df=1.0,max_features=1024, ngram_range=(1,2))
vec = vectorizer.fit(trainClean)

X_train = vec.transform(trainClean)
X_test = vec.transform(testClean)
y_train = train['intent']
y_test = test['intent']


# In[345]:



neigh = KNeighborsClassifier(n_neighbors=5, weights="distance", p=2)
neigh_train = neigh.fit(X_train, y_train) 
y_pred = neigh_train.predict(X_test)

print("Multi-class accuracy:",accuracy_score(y_test, y_pred),"\n")
print(classification_report(y_test, y_pred))


# In[346]:


clf = GaussianNB()
clf.fit(X_train.toarray(),y_train)
y_pred = clf.predict(X_test.toarray())

print("Multi-class accuracy:",accuracy_score(y_test, y_pred),"\n")
print(classification_report(y_test, y_pred))


# In[347]:


clf = SVC(kernel="linear", C=10)

clf.fit(X_train.toarray(),y_train)
y_pred = clf.predict(X_test.toarray())

print("Multi-class accuracy:",accuracy_score(y_test, y_pred),"\n")
print(classification_report(y_test, y_pred))


# In[348]:


clf = ExtraTreesClassifier(n_estimators=200)
clf.fit(X_train.toarray(),y_train)
y_pred = clf.predict(X_test.toarray())

print("Multi-class accuracy:",accuracy_score(y_test, y_pred),"\n")
print(classification_report(y_test, y_pred))


# In[349]:


clf = RandomForestClassifier(n_estimators=200)

clf.fit(X_train.toarray(),y_train)
y_pred = clf.predict(X_test.toarray())

print("Multi-class accuracy:",accuracy_score(y_test, y_pred),"\n")
print(classification_report(y_test, y_pred))


# In[350]:


Models = [SVC(kernel="linear", C=10),RandomForestClassifier(n_estimators=200),ExtraTreesClassifier(n_estimators=200),GaussianNB(),KNeighborsClassifier(n_neighbors=5, weights="distance", p=2)]
#create table to compare Model metric
Models_columns = ['Model Name', 'Accuracy score']
Models_compare = pd.DataFrame(columns = Models_columns)
row_index = 0
for alg in Models:

    #set name and parameters
    Models_name = alg.__class__.__name__
    Models_compare.loc[row_index, 'Model Name'] = Models_name
   #score model with cross validation: 
    alg.fit(X_train.toarray(),y_train)
    y_pred = alg.predict(X_test.toarray())
    Models_compare.loc[row_index, 'Accuracy score'] = accuracy_score(y_test,y_pred)  
    row_index+=1


# In[351]:


Models_compare.sort_values(['Accuracy score'])


# In[352]:



from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=7)

clf = SVC(kernel="linear", C=10)
Multi_class_accuracy=[]
for train_index, test_index in skf.split(X_train, y_train):
    X_train_k, X_test_k = X_train[train_index], X_train[test_index]
    y_train_k, y_test_k = train["intent"][train_index], train["intent"][test_index]
    
    clf.fit(X_train_k,y_train_k)
    y_pred = clf.predict(X_test_k)
    print("Multi-class accuracy:",accuracy_score(y_test_k, y_pred),"\n")
    Multi_class_accuracy.append(accuracy_score(y_test_k, y_pred))


# In[353]:


max(Multi_class_accuracy)

