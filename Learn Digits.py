#!/usr/bin/env python
# coding: utf-8

# In[65]:


from sklearn import datasets, neighbors, linear_model

digits = datasets.load_digits()

#EXPLANATION:
# import datasets, neighbors, linear_model from scikit-learn and load digits dataset

#In general, a learning problem considers a set of n samples of data and then tries to predict 
#properties of unknown data. If each sample is more than a single number and, for instance, a multi-dimensional 
#entry (aka multivariate data), it is said to have several attributes or features.


# In[66]:


type(digits)


# In[67]:


print(digits.DESCR)

#EXPLANATION:
# prints the description of the dataset


# In[68]:


print(digits.data.shape) 

#EXPLANATION:
# number of samples, number of attributes


# In[69]:


import matplotlib.pyplot as plt 

plt.gray() 
plt.matshow(digits.images[8]) 
plt.show() 

#OBSERVATION:
#image shows the dataset of digit 8. Similarily it can be modified to see other digits from 0 to 9.

#Plot histogram
plt.hist(digits.target)
plt.ylabel('Number of instances')
plt.xlabel('digits class')
plt.xticks(range(len(digits.target_names)), digits.target_names);

#OBSERVATION:
#histogram x-axis shows bins and y-axis shows the counts
#for instance, in our dataset, we have 175 times the digit '0'.
#As observed from histogram, the dataset is balanced, having around 175 handwritten digit samples for each digit
#from 0 to 9.


# In[70]:


print(digits.data)

print(digits.target)

#EXPLANATION:
#digits.data gives access to the features that can be used to classify the digits samples
#digits.target gives the ground truth for the digit dataset, that is the number corresponding 
#to each digit image that we are trying to learn


# In[71]:


# library for displaying plots
import matplotlib.pyplot as plt
# display plots in the notebook 
get_ipython().run_line_magic('matplotlib', 'inline')

## First, we repeat the load and preprocessing steps

# Load data
from sklearn import datasets
digits = datasets.load_digits()

# Training and test spliting
from sklearn.model_selection import train_test_split

x_digits, y_digits = digits.data, digits.target
# Test set will be the 25% taken randomly
x_train, x_test, y_train, y_test = train_test_split(x_digits, y_digits, test_size=0.25, random_state=33)

# Preprocess: normalize
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#EXPLANATION:
#We split the entire dataset in two sections. One for training the model and another for testing the model.
#We are training  a model against 75% of the entire dataset and 
#will test it's accuracy against the rest of the 25% dataset.


# In[74]:


from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Create kNN model
model = KNeighborsClassifier(n_neighbors=10)

# Train the model using the training sets
model.fit(x_train, y_train) 

print("Prediction training dataset", model.predict(x_train))
print("Expected training dataset", y_train)

print("Prediction test dataset", model.predict(x_test))
print("Expected test dataset", y_test)

#OBSERVATION:
#Outputs the predicted and expected values of training and test dataset.


# In[75]:


# Evaluate Accuracy in training
from sklearn import metrics
y_train_pred = model.predict(x_train)
print("Accuracy in training", metrics.accuracy_score(y_train, y_train_pred))

#Now we evaluate error in testing
y_test_pred = model.predict(x_test)
print("Accuracy in testing ", metrics.accuracy_score(y_test, y_test_pred))

#OBSERVATION:
#Using KNN model, without tuning any features, the accuracy is too high. This model seems to be overfitting.


# In[76]:


#For evaluating KNN classification algorithm, we calculate three metrics: precision, recall and F1-score
print(metrics.classification_report(y_test, y_test_pred))

#OBSERVATION:
#It can be seen from the below table that digit 0 has true positives in all cases as it's precision and recall is 1.
#Recognition of digit 8 has the lowest performance (f1-score). It's precision is 0.95 and recall is 0.93. 

#Explanation of the below table:
#Precision: This computes the proportion of instances predicted as positives that were correctly evaluated 
#(it measures how right our classifier is when it says that an instance is positive).
#Recall: This counts the proportion of positive instances that were correctly evaluated (measuring how right 
#our classifier is when faced with a positive instance).
#F1-score: This is the harmonic mean of precision and recall, and tries to combine both in a single number.


# In[77]:


#confusion matrix
#It is also known as an error matrix and allows visualization of the performance of an algorithm.

print(metrics.confusion_matrix(y_test, y_test_pred)) 


# In[78]:


#confusion matrix provides a numeric matrix. Transfomed that into a report to understand clearly.
import pandas as pd
print(pd.crosstab(y_test, y_test_pred, rownames=['True'], colnames=['Predicted'], margins=True))


#EXPLANATION:
#non digonal elements, example: 1 of eigth's were misqualified as 3 and 3 times as 1 where as zero was not misqualified.
#diagonal elements show the correct classification of each element.
#total number of classification for each element is mentioned in "All"

#OBSERVATION: As seen from classification report and confusion matrix proves it that 0 can be classified perfectly, where as 8 had the most number of misclassifications.


# In[79]:


#Accuracy of KNN model by tuning the value of k
k_range = range(3, 10)
accuracy = []
for k in k_range:
    m = KNeighborsClassifier(k)
    m.fit(x_train, y_train)
    y_test_pred = m.predict(x_test)
    accuracy.append(metrics.accuracy_score(y_test, y_test_pred))
plt.plot(k_range, accuracy)
plt.xlabel('k value')
plt.ylabel('Accuracy')


# In[80]:


from sklearn.tree import DecisionTreeClassifier
import numpy as np

from sklearn import tree

max_depth=9
random_state=33

# Create decision tree model
model = tree.DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)

# Train the model using the training sets
model.fit(x_train, y_train) 

print("Prediction training dataset", model.predict(x_train))
print("Expected training dataset", y_train)

print("Prediction test dataset", model.predict(x_test))
print("Expected test dataset", y_test)

print("Predicted probabilities", model.predict_proba(x_train[:10]))

# Evaluate Accuracy in training
from sklearn import metrics
y_train_pred = model.predict(x_train)
print("Accuracy in training", metrics.accuracy_score(y_train, y_train_pred))

#Now we evaluate error in testing
y_test_pred = model.predict(x_test)
print("Accuracy in testing ", metrics.accuracy_score(y_test, y_test_pred))


print("Digits:", digits.target)
print("Feature importance:", model.feature_importances_)


# In[81]:


#For evaluating Decision Tree classification algorithm
print(metrics.classification_report(y_test, y_test_pred))

import pandas as pd

#confusion matrix
print(metrics.confusion_matrix(y_test, y_test_pred)) 

#confusion matrix provides a numeric matrix. Transfomed that into a report to understand clearly.
print(pd.crosstab(y_test, y_test_pred, rownames=['True'], colnames=['Predicted'], margins=True))

#OBSERVATION:
# Digits 0 and 6 can be classified much better as compared to digits 8 and 9.


# In[82]:


#In order to avoid bias in the training and testing dataset partition, it is recommended to use k-fold validation.
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# create a composite estimator
model = Pipeline([
        ('scaler', StandardScaler()),
        ('ds', DecisionTreeClassifier())
])

# Fit the model
model.fit(x_train, y_train) 

# create a k-fold cross validation iterator of k=10 folds
cv = KFold(10, shuffle=True, random_state=33)

# by default the score used is the one returned by score method of the estimator (accuracy)
scores = cross_val_score(model, x_digits, y_digits, cv=cv)
print(scores)

from scipy.stats import sem
def mean_score(scores):
    return ("Mean score: {0:.3f} (+/- {1:.3f})").format(np.mean(scores), sem(scores))
print(mean_score(scores))

# Evaluate Accuracy in training
from sklearn import metrics
y_train_pred = model.predict(x_train)
print("Accuracy in training", metrics.accuracy_score(y_train, y_train_pred))

# Now we evaluate error in testing
y_test_pred = model.predict(x_test)
print("Accuracy in testing ", metrics.accuracy_score(y_test, y_test_pred))

#EXPLANATION:
#Python objects that implements fit and predicts are estimators for classification in scikit
#The preprocessing module further provides a utility class StandardScaler that implements the Transformer API 
#to compute the mean and standard deviation on a training set so as to be able to later reapply the same 
#transformation on the testing set. 

#classification: samples belong to two or more classes and we want to learn from already labeled data 
#how to predict the class of unlabeled data.

#sem is to calculate the standard error of the mean (or standard error of measurement) of the 
#values in the input array.

#3fold cross validation
#{0:.3f} -represents mean score, first element of the tuple, with 3 decimal digits
# (+/- {1:.3f}) represents error in the mean score, second element of the tuple, with 3 decimal digits


# In[83]:


#When we use a Pipeline, every chained estimator is stored in the dictionary named_steps and as a list in steps.
print("Model named_steps ",model.named_steps)


# In[84]:


print("Model steps ",model.steps)


# In[85]:


print("param keys of model ",model.get_params().keys())


# In[86]:


print("params of model ",model.get_params())


# In[87]:


#OPTIMIZATION

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import numpy as np

param_grid = {'max_depth': np.arange(3, 10)} 

gs = GridSearchCV(DecisionTreeClassifier(), param_grid)

gs.fit(x_train, y_train)

# summarize the results of the grid search
print("Best score: ", gs.best_score_)
print("Best params: ", gs.best_params_)

#EXPLANATION:
#Changing manually the parameters to find their optimal values is not practical. 
#Instead, we can consider to find the optimal value of the parameters as an optimization problem.
#The sklearn provides an object that, given data, computes the score during the fit of an estimator on a 
#parameter grid and chooses the parameters to maximize the cross-validation score.


# In[88]:


# We print the score for each value of max_depth
for i, max_depth in enumerate(gs.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (gs.cv_results_['mean_test_score'][i],
                                        gs.cv_results_['std_test_score'][i] * 2,
                                        max_depth))


# In[89]:


# create a composite estimator
model = Pipeline([
        ('scaler', StandardScaler()),
        ('ds', DecisionTreeClassifier(max_depth=6))
])

# Fit the model
model.fit(x_train, y_train) 

# create a k-fold cross validation iterator of k=10 folds
cv = KFold(10, shuffle=True, random_state=33)

# by default the score used is the one returned by score method of the estimator (accuracy)
scores = cross_val_score(model, x_digits, y_digits, cv=cv)
def mean_score(scores):
    return ("Mean score: {0:.3f} (+/- {1:.3f})").format(np.mean(scores), sem(scores))
print(mean_score(scores))


# Evaluate Accuracy in training
from sklearn import metrics
y_train_pred = model.predict(x_train)
print("Accuracy in training", metrics.accuracy_score(y_train, y_train_pred))

# Now we evaluate error in testing
y_test_pred = model.predict(x_test)
print("Accuracy in testing ", metrics.accuracy_score(y_test, y_test_pred))


# In[90]:


# Set the parameters by cross-validation

from sklearn.metrics import classification_report

# set of parameters to test
tuned_parameters = [{'max_depth': np.arange(3, 10),
                     'criterion': ['gini', 'entropy'], 
                     'splitter': ['best', 'random'],
                     'class_weight':['balanced', None],
                     'max_leaf_nodes': [None, 5, 10, 20]
                    }]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    # cv = the fold of the cross-validation cv, defaulted to 5
    gs = GridSearchCV(DecisionTreeClassifier(), tuned_parameters, cv=10, scoring='%s_weighted' % score)
    gs.fit(x_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(gs.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    for params, mean_score, scores in gs.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, gs.predict(x_test)
    print(classification_report(y_true, y_pred))
    print()


# In[91]:


# create a composite estimator
model = Pipeline([
        ('scaler', StandardScaler()),
        ('ds', DecisionTreeClassifier(max_leaf_nodes=20, criterion='gini', 
                                      splitter='random', class_weight='balanced', max_depth=3))
])

# Fit the model
model.fit(x_train, y_train) 

# create a k-fold cross validation iterator of k=10 folds
cv = KFold(10, shuffle=True, random_state=33)

# by default the score used is the one returned by score method of the estimator (accuracy)
scores = cross_val_score(model, x_digits, y_digits, cv=cv)
def mean_score(scores):
    return ("Mean score: {0:.3f} (+/- {1:.3f})").format(np.mean(scores), sem(scores))
print(mean_score(scores))

# Evaluate Accuracy in training
from sklearn import metrics
y_train_pred = model.predict(x_train)
print("Accuracy in training", metrics.accuracy_score(y_train, y_train_pred))

# Now we evaluate error in testing
y_test_pred = model.predict(x_test)
print("Accuracy in testing ", metrics.accuracy_score(y_test, y_test_pred))


# In[92]:


# create a composite estimator
model = Pipeline([
        ('scaler', StandardScaler()),
        ('ds', DecisionTreeClassifier(max_leaf_nodes=None, criterion='gini', 
                                      splitter='best', class_weight='balanced', max_depth=9))
])

# Fit the model
model.fit(x_train, y_train) 

# create a k-fold cross validation iterator of k=10 folds
cv = KFold(10, shuffle=True, random_state=33)

# by default the score used is the one returned by score method of the estimator (accuracy)
scores = cross_val_score(model, x_digits, y_digits, cv=cv)
def mean_score(scores):
    return ("Mean score: {0:.3f} (+/- {1:.3f})").format(np.mean(scores), sem(scores))
print(mean_score(scores))

# Evaluate Accuracy in training
from sklearn import metrics
y_train_pred = model.predict(x_train)
print("Accuracy in training", metrics.accuracy_score(y_train, y_train_pred))

# Now we evaluate error in testing
y_test_pred = model.predict(x_test)
print("Accuracy in testing ", metrics.accuracy_score(y_test, y_test_pred))


# In[93]:


# save model
from sklearn.externals import joblib
joblib.dump(model, 'model.pkl') 


# In[34]:


#STEPS FOLLOWED AND EXPLANATION:
#NOTE: Numbers in the results presented below might vary after every execution of the code. Since, the dataset is randomly split into testing and training dataset everytime we load the dataset and excute the code.

#1. Import and load dataset
#2. Plot histogram and draw image (visualisation of dataset)

#3. Split dataset into two sections, training and testing dataset 
#   Testing dataset is 25% of overall dataset taken randomly.

#4. Train the KNN model (fit and predict); KNeighborsClassifier with k_neighbors = 10 
#   Results:
#   Accuracy in training 0.9784706755753526
#   Accuracy in testing  0.9711111111111111
#   OBSERVATION: The classification matrix where precision, recall and F1-score has total average of 0.97. 
#   Tuning the value of 'k' still shows that model seems to be overfitting for some values of 'k'.

#5. Train the Decision Tree model (fit and predict); DecisionTreeClassifier with max_depth=9 and random_state=33
#   max_depth is randomly chosen integer, also tried changing it to 3 and 6. But with 9 it showed better results.
#   random_state is an integer that is used to initialize the random object used by random number generator. It could 0 or 1 or any integer. 
#   I have used integer 33 for random_state throughout this assignment simply to maintain consistency.
#   Results:
#   Accuracy in training 0.9740163325909429
#   Accuracy in testing  0.8555555555555555
#   OBSERVATION: The classification matrix where precision, recall and F1-score has total average of 0.82. This model seems to be better than previous KNN model.
#   Therefore, I have further decided to proceed with Decision Tree classifier and tune the hyperparameters of this model.

#6. Create composite estimator for K-fold cross validation with Pipeline and DecisionTreeClassifier()
#   for cross validation (cv) consider k=10 and random_state=33
#   Results:
#   Mean score: 0.857 (+/- 0.007)
#   Accuracy in training 1.0
#   Accuracy in testing  0.8488888888888889
#   OBSERVATION: Pipeline is for encapsulation to be able to call fit and predict only once on the data to fit a whole sequence of estimators.

#7. Optimization by tuning hyperparameters
#   use GridSearchCV with 'max_depth' in the range of 3 to 10.
#   Results:
#   Best score:  0.8017817371937639
#   Best params:  {'max_depth': 9}
#   OBSERVATION: To avoid biasing in training and test dataset partition for the model, GridSearchCV is used.
#   Changing manually the parameters to find their optimal values is not practical. 
#   Instead, we can consider to find the optimal value of the parameters as an optimization problem.

#8. Create composite estimator for K-fold cross validation with Pipeline and DecisionTreeClassifier(max_depth=6)
#   for cross validation (cv) consider k=10 and random_state=33
#   Results:
#   Mean score: 0.762 (+/- 0.010)
#   Accuracy in training 0.8337045285820341
#   Accuracy in testing  0.7777777777777778
#   OBSERVATION: Clearly, the accuracy is low for training. As a consequence, accuracy is low for testing as well.

#9. Tune more hypermparameters to know the best values to obtain the most optimal model
#   'max_depth': np.arange(3, 10),'criterion': ['gini', 'entropy'], 'splitter': ['best', 'random'], 'class_weight':['balanced', None],'max_leaf_nodes': [None, 5, 10, 20]

#   Results:
#   Best parameters set found on development set:
#   {'class_weight': 'balanced', 'criterion': 'entropy', 'max_depth': 9, 'max_leaf_nodes': None, 'splitter': 'random'}
#   Mean score: 0.870 (+/-0.055)
#   Accuracy in training 0.991833704528582
#   Accuracy in testing  0.8355555555555556

#   OBSERVATION: This is the OPTIMAL model with above values for hyperparameters for the handwritten digits dataset. 
#   It provides the BEST mean score, accuracy in training and testing dataset without overfitting or underfitting.

# Further experiments by modifying some of the hyperparameters for Decision Tree Classifier:

#10.Create composite estimator for K-fold cross validation with Pipeline and DecisionTreeClassifier; 
#   hyperparameters as max_leaf_nodes=20, criterion='gini', splitter='random', class_weight='balanced', max_depth=3
#   Results:
#   Mean score: 0.609 (+/- 0.016)
#   Accuracy in training 0.5775798069784707
#   Accuracy in testing  0.5488888888888889
#   OBSERVATION: Accuracy is lowest with these values of hyperparameters

#11.Create composite estimator for K-fold cross validation with Pipeline and DecisionTreeClassifier; 
#   hyperparameters as max_leaf_nodes=None, criterion='gini', splitter='best', class_weight='balanced', max_depth=9
#   Results:
#   Mean score: 0.851 (+/- 0.007)
#   Accuracy in training 0.9703043801039347
#   Accuracy in testing  0.8266666666666667
#   OBSERVATION: Accuracy is improved and satisfactory model is obtained but not the best (Step 9 shows the best).

#12.Save the model


# In[ ]:


#   FINAL OBSERVATION:
#   KNN model seems to be overfitting despite tuning hyperparameter 'k'.
#   Decision tree model is satisfactory when we tune the hyperparameters (especially max_depth) and we obtain optimal model (Step 9).

#   References
#   https://scikit-learn.org/ (majorly)
#   Classroom slides provided by Prof. Carlos
#   https://stats.stackexchange.com/questions/95209/how-can-i-interpret-sklearn-confusion-matrix

