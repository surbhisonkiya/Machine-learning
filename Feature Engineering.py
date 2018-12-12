#!/usr/bin/env python
# coding: utf-8

# ![](images/EscUpmPolit_p.gif "UPM")

# # Course Notes for Learning Intelligent Systems

# Department of Telematic Engineering Systems, Universidad Politécnica de Madrid, © 2016 Carlos A. Iglesias

# ## [Introduction to Machine Learning II](3_0_0_Intro_ML_2.ipynb)

# # Exercise - The Titanic Dataset

# In this exercise we are going to put in practice what we have learnt in the notebooks of the session. 
# 
# Answer directly in your copy of the exercise and submit it as a moodle task.

# In[2]:


import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
sns.set(color_codes=True)

# if matplotlib is not set inline, you will not see plots
get_ipython().run_line_magic('matplotlib', 'inline')


# # Reading Data

# Assign the variable *df* a Dataframe with the Titanic Dataset from the URL https://raw.githubusercontent.com/gsi-upm/sitc/master/ml2/data-titanic/train.csv"
# 
# Print *df*.

# In[3]:


url = "https://raw.githubusercontent.com/gsi-upm/sitc/master/ml2/data-titanic/train.csv"
df = pd.read_csv(url)
df
# Show the first 5 rows
#df[:5]
#df.head()


# # Munging and Exploratory visualisation

# Obtain number of passengers and features of the dataset

# In[4]:


#df.info()
#Number of samples and features
print("Number of passengers and features:",df.shape)
#Number of passengers, returns the count of non-null rows for PassengerId column.
print("Total number of passengers:", df['PassengerId'].count())


# Obtain general statistics (count, mean, std, min, max, 25%, 50%, 75%) about the column Age

# In[5]:


# General description of the dataset
# print(df.describe()) # to print general statistics for all features
print(df.describe().Age[['count','mean', 'std', 'min', 'max','25%','50%','75%']])

# Another way to achieve the same task
# df['Age'].describe()


# Obtain the median of the age of the passengers

# In[6]:


print("Median age of the passengers:", df['Age'].median())


# Obtain number of missing values per feature

# In[7]:


df.isnull().sum()


# How many passsengers have survived? List them grouped by Sex and Pclass.
# 
# Assign the result to a variable df_1 and print it

# In[8]:


df_1 = df.query('Survived == 1').groupby(['Pclass','Sex']).size()
df_1


# Visualise df_1 as an histogram.

# In[9]:


# Analize distributon
df_1.plot(kind='hist', title = 'Histogram of passengers survived grouped by Pclass and Sex')
plt.show()

# the histogram was not so clear so there is also a bar graph to better understand the visualization of variable df_1.

df_1.plot(kind='bar', title = 'Count of passengers survived grouped by Pclass and Sex')
plt.show()

df_1.hist()


# In[ ]:





# # Feature Engineering

# Here you can find some features that have been proposed for this dataset. Your task is to analyse them and provide some insights. 
# 
# Use pandas and visualisation to justify your conclusions

# ## Feature FamilySize 

# Regarding SbSp and Parch, we can define a new feature, 'FamilySize' that is the combination of both.

# In[342]:


df['FamilySize'] = df['SibSp'] + df['Parch']
df


# In[343]:


df['FamilySize'].hist()
plt.title('Histogram for Feature FamilySize')
plt.xlabel('FamilySize')
plt.show()

df_new = (df[(df.FamilySize==0)].count())
print("Total no of passengers who travelled alone: ", df_new['PassengerId'])


# #### Insights
# 
# The above graph for Feature 'FamilySize' shows that over 500 passengers travelled alone. 
# This can be confirmed by using 'count' function on passengerId column for rows where FamilySize is zero. This count shows that there were 537 passengers who travelled alone out of 891 total passengers.

# ## Feature Alone

# It seems many people who went alone survived. We can define a new feature 'Alone'

# In[344]:


df['Alone'] = (df.FamilySize == 0)
df


# In[345]:


# Added new feature 'Alone_New' to map 0 or 1 value
df['Alone_New'] = df['Alone'].map(lambda x: 1 if x == False else 0)
df.head()


# In[346]:


#We can observe the detail for passengers who went alone and survived
aloneSurvived = df.groupby(['Alone','Survived'])['Survived'].count()
aloneSurvived.plot(kind='bar')
plt.show()


# #### Insights
# 
# The above graph for feature 'Alone' disagrees with the statement made at the start of this feature ("It seems many people who went alone survived. We can define a new feature 'Alone'"). The graph shows that over 350 passengers who went alone died out of total 537 passengers who went alone.
# 
# To confirm this, let's see the plot of who survived the most i.e. passengers with familysize more or less than 1 or equal to 1 since the graph for feature 'Alone' shows that most of the passengers who went alone did not survive.
# 
# The following graph shows that passengers whose 'FamilySize' = 3 survived the most. 

# In[347]:


df.groupby('FamilySize').Survived.mean().plot()
plt.title('Survival mean of FamilySize')
plt.show()


# ## Feature Salutation

# If we observe well in the name variable, there is a 'title' (Mr., Miss., Mrs.). We can add a feature wit this title.

# In[348]:


#Taken from http://www.analyticsvidhya.com/blog/2014/09/data-munging-python-using-pandas-baby-steps-python/
def name_extract(word):
    return word.split(',')[1].split('.')[0].strip()

df['Salutation'] = df['Name'].apply(name_extract)
df


# We can list the different salutations.

# In[349]:


df['Salutation'].unique()


# In[350]:


df.groupby(['Salutation']).size()


# In[351]:


# Distribution
colors_sex = ['#ff69b4', 'b', 'r', 'y', 'm', 'c']
df.groupby('Salutation').size().plot(kind='bar', color=colors_sex)
plt.show()


# There only 4 main salutations, so we combine the rest of salutations in 'Others'.

# In[352]:


def group_salutation(old_salutation):
    if old_salutation == 'Mr':
        return('Mr')
    else:
        if old_salutation == 'Mrs':
            return('Mrs')
        else:
            if old_salutation == 'Master':
                return('Master')
            else: 
                if old_salutation == 'Miss':
                    return('Miss')
                else:
                    return('Others')
df['Salutation'] = df['Salutation'].apply(group_salutation)
df.groupby(['Salutation']).size()


# In[353]:


# Distribution
colors_sex = ['#ff69b4', 'b', 'r', 'y', 'm', 'c']
df.groupby('Salutation').size().plot(kind='bar', color=colors_sex)


# In[354]:


df.boxplot(column='Age', by = 'Salutation', sym='k.')


# #### Insights
# 
# The bar plot shows that over 500 passengers out of total 891 passengers were men (salutation = 'Mr'). The box plot shows the age of most number of male ('Mr') passengers was from 20-40 years with a median value of 30 years.
# 
# The below bar graph shows the survival numbers for each of the salutation. It confirms that most number of passengers who survived were females (Miss and Mrs both inclusive).

# In[355]:


salutationSurvived = df.groupby(['Salutation', 'Survived'])['Survived'].count()
#print(salutationSurvived)
salutationSurvived.plot(kind='bar') #Female survives the most
plt.show()


# ## Features Children and Female

# In[356]:


# Specific features for Children and Female since there are more survivors
df['Children']   = df['Age'].map(lambda x: 1 if x < 6.0 else 0)
df['Female']     = df['Sex'].map(lambda x: 1 if x == "female" else 0)
df


# In[357]:


df['Children'].hist()
plt.title ('Histogram for Feature Children')
plt.show() #less than 50 children
df['Female'].hist()
plt.title ('Histogram for Feature Female')
plt.show() # around 300 females


# #### Insights
# 
# The histogram above shows there are less than 50 children and around 300 females among the total of 891 passengers.

# In[358]:


#Mean of survival for young and female
print("Mean of survival for Feature Children: ", df[df.Children == 1]['Survived'].mean())
print("Mean of survival for Feature Female: ",df[df.Female == 1]['Survived'].mean())

# We plot the histogram per Children
children = sns.FacetGrid(df, col='Survived')
children.map(plt.hist, "Children", color="steelblue") # Survival plot for Children

#Alternative to Seaborn with matplotlib integrated in pandas
df.hist(column='Female', by='Survived', sharey=True) # Survival plot for Female
plt.xlabel('Female')
plt.show()


# #### Insights
# 
# Above histograms show the survival of Children and Females respectively.
# 
# According to histogram, around 10-15 children died. However, around 30 children survived. That indicates that mojority of total children survived.
# 
# With respect to survival of females, sadly around 100 women died. However, around 200 females survived. This indicates that females who survived is more than females who died.
# 
# Overall, out of 44 Children, 30 survived and out of ~300 women, ~200 of them survived.

# In[359]:


#Also, df.query could be used to get the survival count for Children and Female
#Since the survival of Children histogram doesnot give clear figures, bar grpah is also plotted for the same.
df.query("Children == 1").groupby(['Survived']).size().plot(kind='bar', title = 'Number of Children Survived') # 10-15 children died; over 30 children survived


# ## Feature AgeGroup

# In[360]:


# Group ages to simplify machine learning algorithms.  0: 0-5, 1: 6-10, 2: 11-15, 3: 16-59 and 4: 60-80
df['AgeGroup'] = 0
df.loc[(df.Age<6),'AgeGroup'] = 0
df.loc[(df.Age>=6) & (df.Age < 11),'AgeGroup'] = 1
df.loc[(df.Age>=11) & (df.Age < 16),'AgeGroup'] = 2
df.loc[(df.Age>=16) & (df.Age < 60),'AgeGroup'] = 3
df.loc[(df.Age>=60),'AgeGroup'] = 4
df


# In[361]:


# Distribution
colors_sex = ['#ff69b4', 'b', 'r', 'y', 'm', 'c']
df.groupby('AgeGroup').size().plot(kind='bar', color=colors_sex)
plt.title('Size of AgeGroup')
plt.show()


# #### Insights
# 
# Around 600 passengers out of total 891 passengers are from AgeGroup 3, i.e. of age between 16-60 years.

# In[362]:


#Survival of AgeGroups
df.groupby(['AgeGroup']).Survived.mean().plot(kind='bar')
plt.title('Survival mean of AgeGroup')
plt.show()


# #### Insights
# 
# Passengers from AgeGroup 2 i.e. of age between 11-16 years has the highest survival rate as compared to the other AgeGroups (over 55% of total passengers of AgeGroup 2 survived).

# ## Feature Deck
# Only 1st class passengers have cabins, the rest are ‘Unknown’. A cabin number looks like ‘C123’. The letter refers to the deck.

# In[363]:


def substrings_in_string(big_string, substrings):
    if type(big_string) == float:
        if np.isnan(big_string):
            return 'X'
    for substring in substrings:
        if big_string.find(substring) != -1:
            return substring
    print(big_string)
    return 'X'
 
#Turning cabin number into Deck
cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']
df['Deck']=df['Cabin'].map(lambda x: substrings_in_string(x, cabin_list))
df


# In[364]:


# Distribution
colors_sex = ['#ff69b4', 'b', 'r', 'y', 'm', 'c']
df.groupby('Deck').size().plot(kind='bar', color=colors_sex)
plt.title('Size of Deck')
plt.show()


# #### Insights
# 
# The bar graph for Feature 'Deck'  shows that over 650 passengers did not travel in 1st class cabins.

# In[365]:


#Survival of deck
df.groupby(['Deck']).Survived.mean().plot(kind='bar')
plt.title('Survival mean of Deck')
plt.show()


# #### Insights
# 
# The above bar graph shows that decks 'B', 'D' and 'E' had almost the same and highest survival rate i.e. more than 70%. Followed by deck 'C' and 'F' both having survival rate of 55-60%. Deck 'G' and 'A' had 50% and 45-50% survival rate respectively.
# 
# Deck 'X', the deck that is not considered as 1st class had the least survival rate i.e. 30%. It shows that 70% of passengers who were not travelling in 1st class died.

# ## Feature FarePerPerson

# This feature is created from two previous features: Fare and FamilySize.

# In[366]:


df['FarePerPerson']= df['Fare'] / (df['FamilySize'] + 1)
df


# In[367]:


df['FarePerPerson'].plot(kind='kde', title = 'Fare Per Person')
plt.show()

df.groupby([df.FarePerPerson<100]).Survived.mean().plot(kind='bar', title = ' Survival mean of passengers with fare < 100')
plt.show()

#print(df[(df.FarePerPerson<500) & (df.FarePerPerson>200)].count())


# #### Insights
# 
# The first graph which is the KDE (Kernel Density Estimation) plot, shows that maximum number of passengers had payed fares between 0-50 and some of them upto 100 units. Very few passengers payed fares over 100 units.
# 
# The second bar graph shows the survival mean of passengers who payed fare less than 100 units. Around 38% of them survived. However, majority of them who payed fares less than 100 units couldnot survive.
# 
# Let's further divide fares from 0-50 and 50-100 units to get more accurate statistics.

# In[368]:


FarePerPersonSurvived = df.groupby([df.FarePerPerson<50,'Survived'])['Survived'].count()
FarePerPersonSurvived.plot(kind='bar')

FarePerPersonSurvived = df.groupby([(df.FarePerPerson>50) & (df.FarePerPerson<100),'Survived'])['Survived'].count()
FarePerPersonSurvived.plot(kind='line')
plt.title('Different ranges of FarePerPerson and their respective survival')
plt.show()


# #### Insights
# 
# The bar plot shows around 300 passengers who payed fares less than 50 units survived. However, over 500 of them died.
# 
# The line plot shows the count of passengers who payed fares between 50-100 units and their survival. It can be observed that passengers falling into this category mostly survived. Also, the survival of passengers in this category is higher than the count of dead passengers who payed fares between 50-100 units.
# 
# Overall, the majority of the passengers who could not survive are the ones who had fares less than 50 units.

# ## Feature AgeClass

# Since age and class are both numbers we can just multiply them and get a new feature.
# 

# In[369]:


df['AgeClass']=df['Age']*df['Pclass']
df
#print(df[df.AgeClass<1].count())


# In[370]:


df['AgeClass'].hist()
plt.xlabel('AgeClass')
plt.title('Histogram of Feature AgeClass')
plt.show()# x-axis is value of 'AgeClass', y-axis is count


# #### Insights
# 
# From the above histogram, it can be observed that maximum number of passengers have AgeClass between 50-75. On the broad spectrum, most of the passengers have AgeClass between 0-100. Very few have AgeClass between 100-150 and hardly any between 150-250.

# In[371]:


AgeClassSurvived = df.groupby([df.AgeClass<50,'Survived'])['Survived'].count()#270
AgeClassSurvived.plot(kind='bar')

AgeClassSurvived = df.groupby([(df.AgeClass>50) & (df.AgeClass<100),'Survived'])['Survived'].count()#339
AgeClassSurvived.plot(kind='line')#blue line

AgeClassSurvived = df.groupby([(df.AgeClass>100) & (df.AgeClass<250),'Survived'])['Survived'].count()#89
AgeClassSurvived.plot(kind='line')
plt.title('Different ranges of AgeClass and their respective survival')
plt.show()


# #### Insights
# 
# The above plot shows three divisions of graph. They are:
# 
# 1) The first division is for passengers having AgeClass less than 50. This is represented by the bar graph in the above plot. In this division, survival of passengers is higher. Around 100 passengers died but over 150 passengers survived.
# 
# 2) The second division is for passengers having AgeClass between 50-100. This is represented by the blue line in the above plot. Majority passengers of this category could not survive i.e. over 200 passengers from this division died.
# 
# 3) The third division is for passengers having AgeClass between 100-250. This is represented by the brown/orange line in the above plot. Hardly anyone in this category survived. Sadly, almost everyone in this category died.

# #### References
# 
# 1) https://pandas.pydata.org/pandas-docs/version/0.23/generated/pandas.DataFrame.plot.html
# 
# 2) https://www.analyticsvidhya.com/blog/2014/09/data-munging-python-using-pandas-baby-steps-python/
# 
# 3) https://realpython.com/python-histograms/
# 
# 4) https://towardsdatascience.com/a-guide-to-pandas-and-matplotlib-for-data-exploration-56fad95f951c
# 
# 5) Classroom lectures and notebooks provided by Prof. Carlos, UPM.

# ## Licence

# The notebook is freely licensed under under the [Creative Commons Attribution Share-Alike license](https://creativecommons.org/licenses/by/2.0/).  
# 
# © 2016 Carlos A. Iglesias, Universidad Politécnica de Madrid.
