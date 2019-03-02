# Intent-Detection
AI Intent Detection Task


Step 1 :
I started with reading the train and test file as infile and converted them in a dataframe with one column as message and other as the intent.


Step 2:
Cleaning the data which included removing stopwords, changing all the words to lower case adn stemming and lemmatization.


Step 3:
This step involved feature extraction using TFIDF vectorizer.


Step 4:
In this step 5 models were run. Fitting them on the training sets and predicting them of the
testing set. The metric of measurement here was the accuracy score and the models used are
Support Vector Classifier, Random Forest Classifier, Extra Trees Classifier,Gaussian Naive Bayes, KNN Classifier

Step 5:
The accuracy score of SVC was highest, so this step involved stratified K-Fold Cross Validation method using SVC.
K-Fold Cross Validation is with 7 splits.
