#!/usr/bin/env python
# coding: utf-8

# ![](images/EscUpmPolit_p.gif "UPM")

# # Course Notes for Learning Intelligent Systems

# Department of Telematic Engineering Systems, Universidad Politécnica de Madrid, © 2016 Carlos A. Iglesias

# ## [Introduction to Machine Learning II](3_0_0_Intro_ML_2.ipynb)

# # Exercise 2 - The Titanic Dataset

# In this exercise we are going to put in practice what we have learnt in the notebooks of the session. 
# 
# In the previous notebook we have been applying the SVM machine learning algorithm.
# 
# Your task is to apply other machine learning algorithms (at least 2) that you have seen in theory or others you are interested in.
# 
# You should compare the algorithms and describe your experiments.

# # Load and Clean

# In[2]:


# General import and load data
import pandas as pd
import numpy as np

from pandas import Series, DataFrame
from sklearn import neighbors, linear_model

# Training and test spliting
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


# Estimators
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression



# Evaluation

from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


# Optimization
from sklearn.model_selection import GridSearchCV


# Visualisation
import seaborn as sns
# library for displaying plots
import matplotlib.pyplot as plt
sns.set(color_codes=True)
# display plots in the notebook 
#%matplotlib inline
get_ipython().run_line_magic('run', 'plot_learning_curve')


# In[3]:


#We get a URL with raw content (not HTML one)
url="https://raw.githubusercontent.com/gsi-upm/sitc/master/ml2/data-titanic/train.csv"
df = pd.read_csv(url)
df.head()


#Fill missing values
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Sex'].fillna('male', inplace=True)
df['Embarked'].fillna('S', inplace=True)

# Encode categorical variables
df['Age'] = df['Age'].fillna(df['Age'].median())
df.loc[df["Sex"] == "male", "Sex"] = 0
df.loc[df["Sex"] == "female", "Sex"] = 1
df.loc[df["Embarked"] == "S", "Embarked"] = 0
df.loc[df["Embarked"] == "C", "Embarked"] = 1
df.loc[df["Embarked"] == "Q", "Embarked"] = 2

# Drop colums
df.drop(['Cabin', 'Ticket', 'Name'], axis=1, inplace=True)

#Show proprocessed df
df.head()


# In[4]:


#Check types are numeric
df.dtypes
df['Sex'] = df['Sex'].astype(np.int64)
df['Embarked'] = df['Embarked'].astype(np.int64)
df.dtypes


# In[5]:


#Check there are not missing values
df.isnull().any()


# # Train and Test Splitting
# 
# Split the dataset into two parts. 25% randomly taken data from the dataset forms the testing dataset. Rest of it forms the training dataset. 
# 
# Training dataset is used to train the models and testing dataset is used for prediction from the trained models.

# In[6]:


# Features of the model
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
# Transform dataframe in numpy arrays
x = df[features].values
y = df['Survived'].values


# Test set will be the 25% taken randomly
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=33)

# Preprocess: normalize
scaler = preprocessing.StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# # Define Model, Train and Evaluate
# 
# Create the models and train them using training dataset. Evaluate the accuracy of it's performance using test datatset.
# 
# ### KNN

# In[7]:


#KNN

# Create kNN model
KNNmodel = KNeighborsClassifier(n_neighbors=20)

# Train the model using the training sets
KNNmodel.fit(x_train, y_train) 

# Evaluate Accuracy in training
y_train_pred_1 = KNNmodel.predict(x_train)
print("Accuracy in training", round(metrics.accuracy_score(y_train, y_train_pred_1)*100,2), "%")
y_train_pred_knn = round(metrics.accuracy_score(y_train, y_train_pred_1)*100,2)

#Now we evaluate error in testing
y_test_pred_1 = KNNmodel.predict(x_test)
print("Accuracy in testing ", round(metrics.accuracy_score(y_test, y_test_pred_1)*100,2), "%")
y_test_pred_knn = round(metrics.accuracy_score(y_test, y_test_pred_1)*100,2)


# ### Random Forest Classifier

# In[7]:


#Random Forest Classifier

random_forest = RandomForestClassifier()#n_estimators=10
random_forest.fit(x_train, y_train)

# Evaluate Accuracy in training
y_train_pred_2 = random_forest.predict(x_train)
print("Accuracy in training", round(metrics.accuracy_score(y_train, y_train_pred_2)*100,2), "%")
y_train_pred_random = round(metrics.accuracy_score(y_train, y_train_pred_2)*100,2)

#Now we evaluate error in testing
y_test_pred_2 = random_forest.predict(x_test)
print("Accuracy in testing ", round(metrics.accuracy_score(y_test, y_test_pred_2)*100,2), "%")
y_test_pred_random = round(metrics.accuracy_score(y_test, y_test_pred_2)*100,2)



# ### Decision Tree Classifier

# In[8]:


#Decision Tree Classifier
decision_tree = DecisionTreeClassifier() 
decision_tree.fit(x_train, y_train)  

# Evaluate Accuracy in training
y_train_pred_3 = decision_tree.predict(x_train)
print("Accuracy in training", round(metrics.accuracy_score(y_train, y_train_pred_3)*100,2), "%")
y_train_pred_decision= round(metrics.accuracy_score(y_train, y_train_pred_3)*100,2)


#Now we evaluate error in testing
y_test_pred_3 = decision_tree.predict(x_test)
print("Accuracy in testing ", round(metrics.accuracy_score(y_test, y_test_pred_3)*100,2), "%")
y_test_pred_decision = round(metrics.accuracy_score(y_test, y_test_pred_3)*100,2)


# ### LinearSVC

# In[9]:


linear_svc = LinearSVC()
linear_svc.fit(x_train, y_train)

# Evaluate Accuracy in training
y_train_pred_4 = linear_svc.predict(x_train)
print("Accuracy in training", round(metrics.accuracy_score(y_train, y_train_pred_4)*100,2), "%")
y_train_pred_linear =  round(metrics.accuracy_score(y_train, y_train_pred_4)*100,2)

#Now we evaluate error in testing
y_test_pred_4 = linear_svc.predict(x_test)
print("Accuracy in testing ", round(metrics.accuracy_score(y_test, y_test_pred_4)*100,2), "%")
y_test_pred_linear = round(metrics.accuracy_score(y_test, y_test_pred_4)*100,2)



# # Comparision without tuning the model
# 
# Below it summarizes the accuracy of performance of different models against the same training and testing dataset.
# 
# ### For Training Dataset

# In[10]:


results = pd.DataFrame({
    'Model': ['Linear SVC', 'KNN', 
              'Random Forest', 'Decision Tree'],
    'Training Score (%)': [y_train_pred_linear,y_train_pred_knn,y_train_pred_random,y_train_pred_decision]})
result_df = results.sort_values(by='Training Score (%)', ascending=False)
result_df = result_df.set_index('Training Score (%)')
result_df.head(9)


# ### For Testing Dataset

# In[11]:


results = pd.DataFrame({
    'Model': ['Linear SVC', 'KNN', 
              'Random Forest', 'Decision Tree'],
    'Testing Score (%)': [y_test_pred_linear,y_test_pred_knn,y_test_pred_random,y_test_pred_decision]})
result_df = results.sort_values(by='Testing Score (%)', ascending=False)
result_df = result_df.set_index('Testing Score (%)')
result_df.head(9)


# # Observations
# 
# Above are the tables for all the models measuring their accuracy for training and testing dataset respectively. 
# 
# With respect to training dataset, it can be observed that accuracy for Random Forest and Decision Tree models is over 95% where as accuracy for KNN and Linear SVC models is almost same (~80%). It could be safe to say that former two models are a bit overfitting models and latter two are a bit underfitting for training dataset. However, accuracy for testing dataset is similar for all four models i.e. ~84%. This percentage seems acceptable for testing dataset. 
# 

# # Null Accuracy
# 
# Performed null accuracy to evaluate the accuracy if the model always predicts the most frequent class.

# In[12]:


# Count number of samples per class
s_y_test = Series(y_test)
s_y_test.value_counts()


# In[13]:


# Mean of ones
y_test.mean()


# In[14]:


# Mean of zeros
1 - y_test.mean() 


# In[15]:


# Calculate null accuracy (binary classification coded as 0/1)
max(y_test.mean(), 1 - y_test.mean())


# In[16]:


# Calculate null accuracy (multiclass classification)
s_y_test.value_counts().head(1) / len(y_test)


# # Observations
# 
# Since the accuracy for all the four models is over 80% (see above tables for accurate percentage), they all are better than the null accuracy.

# # Confusion Matrix and F-score
# 
# Confusion matrix is also known as an error matrix and allows visualization of the performance of an algorithm. F-score is a measure of test's accuracy.
# 
# ### KNN

# In[17]:


print(metrics.confusion_matrix(y_test, y_test_pred_1)) 

print(metrics.classification_report(y_test, y_test_pred_1))


# ### Random Forest Classifier

# In[18]:


print(metrics.confusion_matrix(y_test, y_test_pred_2)) 

print(metrics.classification_report(y_test, y_test_pred_2))


# ### Decision Tree Classifier

# In[19]:


print(metrics.confusion_matrix(y_test, y_test_pred_3)) 

print(metrics.classification_report(y_test, y_test_pred_3))


# ### Linear SVC

# In[20]:


print(metrics.confusion_matrix(y_test, y_test_pred_4)) 

print(metrics.classification_report(y_test, y_test_pred_4))


# # Observations
# 
# As mentioned earlier, accuracy for testing dataset for all four models is almost similar ~84%. 
# 
# This is also proved from the above confusion and F-score matrix. The weighted average for Random Forest Classifier is 85% whereas for KNN and Decision Tree Classifier it is 84% and Linear SVC is 83%.

# # ROC and AUC
# 
# ROC (Receiver Operating Characteristics) and AUC (Area Under The Curve) curves are used to visualise the performance of multi-class classification problem by supplying various thresholds. ROC is a probability curve and AUC tells how much the model is capable to distinguish between classes. Higher the AUC, better is the model at predicting 0s as 0 and 1s as 1.
# 
# ### KNN

# In[21]:


y_pred_prob = KNNmodel.predict_proba(x_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for Titanic')
plt.xlabel('False Positive Rate (1 - Recall)')
plt.xlabel('True Positive Rate (Sensitivity)')
plt.grid(True)


# In[22]:


#Threshold used by the decision function, thresholds[0] is the number of 
thresholds

#Histogram of probability vs actual
dprob = pd.DataFrame(data = {'probability':y_pred_prob, 'actual':y_test})
dprob.probability.hist(by=dprob.actual, sharex=True, sharey=True)

#Function to evaluate thresholds of the ROC curve
def evaluate_threshold(threshold):
    print('Sensitivity:', tpr[thresholds > threshold][-1])
    print('Recall:', 1 - fpr[thresholds > threshold][-1])

    
evaluate_threshold(0.74)

# AUX
print("AUC: ",roc_auc_score(y_train, y_train_pred_1))


# ### Random Forest Classifier

# In[23]:


y_pred_prob = random_forest.predict_proba(x_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for Titanic')
plt.xlabel('False Positive Rate (1 - Recall)')
plt.xlabel('True Positive Rate (Sensitivity)')
plt.grid(True)


# In[24]:


#Threshold used by the decision function, thresholds[0] is the number of 
thresholds

#Histogram of probability vs actual
dprob = pd.DataFrame(data = {'probability':y_pred_prob, 'actual':y_test})
dprob.probability.hist(by=dprob.actual, sharex=True, sharey=True)

#Function to evaluate thresholds of the ROC curve
def evaluate_threshold(threshold):
    print('Sensitivity:', tpr[thresholds > threshold][-1])
    print('Recall:', 1 - fpr[thresholds > threshold][-1])

    
evaluate_threshold(0.74)

# AUX
print("AUC: ",roc_auc_score(y_train, y_train_pred_2))


# ### Decision Tree Classifier

# In[25]:


y_pred_prob = decision_tree.predict_proba(x_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for Titanic')
plt.xlabel('False Positive Rate (1 - Recall)')
plt.xlabel('True Positive Rate (Sensitivity)')
plt.grid(True)


# In[26]:


#Threshold used by the decision function, thresholds[0] is the number of 
thresholds

#Histogram of probability vs actual
dprob = pd.DataFrame(data = {'probability':y_pred_prob, 'actual':y_test})
dprob.probability.hist(by=dprob.actual, sharex=True, sharey=True)

#Function to evaluate thresholds of the ROC curve
def evaluate_threshold(threshold):
    print('Sensitivity:', tpr[thresholds > threshold][-1])
    print('Recall:', 1 - fpr[thresholds > threshold][-1])

    
evaluate_threshold(0.74)

# AUX
print("AUC: ",roc_auc_score(y_train, y_train_pred_3))


# ### Linear SVC

# In[27]:



# Create kNN model
model = SVC(kernel='linear', probability=True, gamma=3.0)
model.fit(x_train, y_train)

y_pred_prob = model.predict_proba(x_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for Titanic')
plt.xlabel('False Positive Rate (1 - Recall)')
plt.xlabel('True Positive Rate (Sensitivity)')
plt.grid(True)


# In[28]:


#Threshold used by the decision function, thresholds[0] is the number of 
thresholds

#Histogram of probability vs actual
dprob = pd.DataFrame(data = {'probability':y_pred_prob, 'actual':y_test})
dprob.probability.hist(by=dprob.actual, sharex=True, sharey=True)

#Function to evaluate thresholds of the ROC curve
def evaluate_threshold(threshold):
    print('Sensitivity:', tpr[thresholds > threshold][-1])
    print('Recall:', 1 - fpr[thresholds > threshold][-1])

    
evaluate_threshold(0.74)

# AUX
print("AUC: ",roc_auc_score(y_train, y_train_pred_4))


# # Observations
# 
# For the ROC curve, probability is plotted against True Positive Ratio (TPR) to False Positive Ratio (FPR). Accuracy for the models as seen from above AUC curves are:
# 
# KNN: 77%; Random Forest: 95%; Decision Tree: 98%; Linear SVC: 77%
# 
# Clearly, Decision Tree and Random Forest models outperforms. Therefore, these two are better models than the other two. However, even KNN and Linear SVC models have decent performance (more than 50%).

# # Train and Evaluate with K-Fold
# 
# In order to avoid bias in the training and testing dataset partition, it is recommended to perform k-fold validation individually for each of the models.
# 
# ### KNN

# In[29]:


#KNN Train and Evaluate with K-Fold

# create a k-fold cross validation iterator of k=10 folds
cv = KFold(10, shuffle=True, random_state=33)

# by default the score used is the one returned by score method of the estimator (accuracy)
scores = cross_val_score(KNNmodel, x, y, cv=cv)
#print(scores)

print("Scores in every iteration", scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

plot_learning_curve(KNNmodel, "Learning curve with K-Fold", x, y, cv=cv)


# ### Random Forest Classifier

# In[30]:


# create a k-fold cross validation iterator of k=10 folds
cv = KFold(10, shuffle=True, random_state=33)

# by default the score used is the one returned by score method of the estimator (accuracy)
scores = cross_val_score(random_forest, x, y, cv=cv)
#print(scores)

print("Scores in every iteration", scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

plot_learning_curve(random_forest, "Learning curve with K-Fold", x, y, cv=cv)


# ### Decision Tree Classifier

# In[31]:


# create a k-fold cross validation iterator of k=10 folds
cv = KFold(10, shuffle=True, random_state=33)

# by default the score used is the one returned by score method of the estimator (accuracy)
scores = cross_val_score(decision_tree, x, y, cv=cv)
#print(scores)

print("Scores in every iteration", scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

plot_learning_curve(decision_tree, "Learning curve with K-Fold", x, y, cv=cv)


# ### Linear SVC

# In[32]:


# create a k-fold cross validation iterator of k=10 folds
cv = KFold(10, shuffle=True, random_state=33)

# by default the score used is the one returned by score method of the estimator (accuracy)
scores = cross_val_score(linear_svc, x, y, cv=cv)
#print(scores)

print("Scores in every iteration", scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

plot_learning_curve(linear_svc, "Learning curve with K-Fold", x, y, cv=cv)


# # Observations
# 
# Even though learning curves for Linear SVC and KNN models look great, yet the accuracy for both of them, with K-Fold validation is very low, summarised below:
# 
# KNN: 71% ; Random Forest: 80% ; Decision Tree: 78% ; Linear SVC: 67%
# 
# 
# Overall, for all four models, accuracy needs to be improved as this is not the best possible performance for any of the models. Therefore, we need to tune the hyperparameter(s) for each of the models.

# # Tuning Hyperparameters
# 
# Find out what are the best values for the hyperparameter(s) for each of the models so that we can obtain satisfactory accuracy while evaluating performance of each of the model.
# 
# ### KNN

# In[33]:


k_range = range(1, 30)
accuracy = []
for k in k_range:
    m = KNeighborsClassifier(k)
    m.fit(x_train, y_train)
    y_test_pred = m.predict(x_test)
    accuracy.append(metrics.accuracy_score(y_test, y_test_pred))
plt.plot(k_range, accuracy)
plt.xlabel('k value')
plt.ylabel('Accuracy')
plt.show()


# ### Random Forest Classifier

# In[34]:


from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}

# Instantiate the grid search model
gs = GridSearchCV(RandomForestClassifier(), param_grid)

gs.fit(x_train, y_train)

# summarize the results of the grid search
print("Best score: ", gs.best_score_)
print("Best params: ", gs.best_params_)


# ### Decision Tree Classifier

# In[35]:


#OPTIMIZATION
param_grid = {'max_depth': np.arange(3, 10),
                     'criterion': ['gini', 'entropy'], 
                     'splitter': ['best', 'random'],
                     'class_weight':['balanced', None],
                     'max_leaf_nodes': [None, 5, 10, 20]
                    } 

gs = GridSearchCV(DecisionTreeClassifier(), param_grid)

gs.fit(x_train, y_train)

# summarize the results of the grid search
print("Best score: ", gs.best_score_)
print("Best params: ", gs.best_params_)

#scores = cross_val_score(model, x, y, cv=cv)
# We print the score for each value of max_depth
#for i, max_depth in enumerate(gs.cv_results_['params']):
 #   print("%0.3f (+/-%0.03f) for %r" % (gs.cv_results_['mean_test_score'][i],
 #                                       gs.cv_results_['std_test_score'][i] * 2,
 #                                       max_depth))


# ### Linear SVC

# In[36]:


#OPTIMIZATION
param_grid = {'C': np.arange(1.0, 10.0)
#              'max_iter': np.arange(1000,2000)
                      
                    } 
gs = GridSearchCV(LinearSVC(), param_grid)
gs.fit(x_train, y_train)

# summarize the results of the grid search
print("Best score: ", gs.best_score_)
print("Best params: ", gs.best_params_)


# # Train and Optimize
# 
# Recreate the model with the best values of hyperparameters obtained above for each of the model and train the model again. Also, evaluate the performance of each model after tuning it.
# 
# ### KNN

# In[54]:


# Create kNN model
KNNmodel = KNeighborsClassifier(n_neighbors=8)

# Train the model using the training sets
KNNmodel.fit(x_train, y_train) 

# Evaluate Accuracy in training
y_train_pred = KNNmodel.predict(x_train)
print("Accuracy in training", round(metrics.accuracy_score(y_train, y_train_pred)*100,2), "%")
y_train_pred_knn = round(metrics.accuracy_score(y_train, y_train_pred)*100,2)

#Now we evaluate error in testing
y_test_pred = KNNmodel.predict(x_test)
print("Accuracy in testing ", round(metrics.accuracy_score(y_test, y_test_pred)*100,2), "%")
y_test_pred_knn = round(metrics.accuracy_score(y_test, y_test_pred)*100,2)


# ### Random Forest Classifier

# In[48]:


#Random Forest Classifier
random_forest = RandomForestClassifier(bootstrap = True, max_depth = 90, max_features = 2, min_samples_leaf = 3, min_samples_split = 8, n_estimators = 100)


#n_estimators=100, max_depth=2,random_state=0

random_forest.fit(x_train, y_train)

# Evaluate Accuracy in training
y_train_pred = random_forest.predict(x_train)
print("Accuracy in training", round(metrics.accuracy_score(y_train, y_train_pred)*100,2), "%")
y_train_pred_random = round(metrics.accuracy_score(y_train, y_train_pred)*100,2)

#Now we evaluate error in testing
y_test_pred = random_forest.predict(x_test)
print("Accuracy in testing ", round(metrics.accuracy_score(y_test, y_test_pred)*100,2), "%")
y_test_pred_random = round(metrics.accuracy_score(y_test, y_test_pred)*100,2)


# ### Decision Tree Classifier

# In[39]:


#Decision Tree Classifier
decision_tree = DecisionTreeClassifier(class_weight= None, criterion= 'gini', max_depth= 3, max_leaf_nodes= None, splitter = 'best') 
decision_tree.fit(x_train, y_train)  

# Evaluate Accuracy in training
y_train_pred = decision_tree.predict(x_train)
print("Accuracy in training", round(metrics.accuracy_score(y_train, y_train_pred)*100,2), "%")
y_train_pred_decision= round(metrics.accuracy_score(y_train, y_train_pred)*100,2)


#Now we evaluate error in testing
y_test_pred = decision_tree.predict(x_test)
print("Accuracy in testing ", round(metrics.accuracy_score(y_test, y_test_pred)*100,2), "%")
y_test_pred_decision = round(metrics.accuracy_score(y_test, y_test_pred)*100,2)


# ### Linear SVC

# In[47]:


linear_svc = LinearSVC(C=5)
linear_svc.fit(x_train, y_train)

# Evaluate Accuracy in training
y_train_pred = linear_svc.predict(x_train)
print("Accuracy in training", round(metrics.accuracy_score(y_train, y_train_pred)*100,2), "%")
y_train_pred_linear =  round(metrics.accuracy_score(y_train, y_train_pred)*100,2)

#Now we evaluate error in testing
y_test_pred = linear_svc.predict(x_test)
print("Accuracy in testing ", round(metrics.accuracy_score(y_test, y_test_pred)*100,2), "%")
y_test_pred_linear = round(metrics.accuracy_score(y_test, y_test_pred)*100,2)


# # Comparision after tuning the model
# 
# Below it summarizes the accuracy of performance of different tuned models against the same training and testing dataset.
# 
# ### For Training Dataset

# In[41]:


results = pd.DataFrame({
    'Model': [ 'Linear SVC','KNN', 
              'Random Forest', 'Decision Tree'],
    'Training Score (%)': [y_train_pred_linear,y_train_pred_knn,y_train_pred_random,y_train_pred_decision]})
result_df = results.sort_values(by='Training Score (%)', ascending=False)
result_df = result_df.set_index('Training Score (%)')
result_df.head(9)


# ### For Testing Dataset

# In[42]:


results = pd.DataFrame({
    'Model': ['Linear SVC','KNN', 
              'Random Forest', 'Decision Tree'],
    'Testing Score (%)': [y_test_pred_linear, y_test_pred_knn,y_test_pred_random,y_test_pred_decision]})
result_df = results.sort_values(by='Testing Score (%)', ascending=False)
result_df = result_df.set_index('Testing Score (%)')
result_df.head(9)


# # Observations
# 
# As it can be observed from the above comparision table, after tuning the models, performance and accuracy percentage has been improved especially for Random Forest Classifier model.
# 
# Random Forest Classifier is the best model out of the four for Titanic dataset. It has training score of ~88% and testing score of ~87%, highest amongst all four models for both training and testing dataset.
# 
# KNN and Decision Tree models are almost similar in their performance rate whereas Linear SVC is the lowest for both training and testing models.

# # Conclusion
# 
# For the titanic dataset, and as mentioned earlier, Random Forest Classifier is the best model out of the chosen four models. It has training score of ~88% and testing score of ~87%, highest amongst all four models for both training and testing dataset.
# 
# Below are the detailed conclusion for each of the models:
# 
# 1) KNN: The accuracy for training dataset increased after tuning the hyperparameters (from 80.84% to 82.63%). However, after tuning, the accuracy for testing dataset is almost the same, rather slightly high after tuning it (85.65%).
# 
# 2) Random Forest Classifier: After tuning the hyperparameters for this model, accuracy has been increased for testing dataset (86.55%) and the model also showed improvement in the performance against the training dataset (from 96.11% to 87.87%). It is not an overfitting model anymore.
# 
# 3) Decision Tree Classifier: Default values of hyperparameter seems good for this model (83.86%). After tuning, the accuracy for testing dataset is almost the same, rather slightly high after tuning it (84.75%).
# 
# 4) Linear SVC: Default values of hyperparameter seems good for this model (83.41%). Despite tuning, the accuracy for testing dataset is almost the same, rather slightly low after tuning it (82.96%).

# #### References
# 
# 1) https://proquest.safaribooksonline.com/book/programming/python/9781783555130
# 
# 2) https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier
# 
# 3) https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
# 
# 4) https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
# 
# 5) https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5
# 
# 6) Classroom lectures, slides and notebooks provided by Prof. Carlos, UPM.

# ## Licence

# The notebook is freely licensed under under the [Creative Commons Attribution Share-Alike license](https://creativecommons.org/licenses/by/2.0/).  
# 
# © 2016 Carlos A. Iglesias, Universidad Politécnica de Madrid.
