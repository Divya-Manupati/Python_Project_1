#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install numpy


# In[2]:


pip install pandas


# In[3]:


pip install seaborn


# In[4]:


pip install matplotlib


# In[5]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# ### 1.Load the data file using the pandas.

# In[6]:


data = pd.read_csv('Dataset of App rating project/googleplaystore.csv') 


# In[7]:


data


# In[8]:


data.head()


# In[9]:


data.tail()


# In[10]:


data.shape


# ### 2) check the null values in the data. Get the number of null values for each column.

# In[11]:


# To check null values in the data.
data.isnull()


# In[12]:


# null values for each column.
data.isnull().sum()


# In[13]:


data.isnull().sum().sum()


# ### 3) Drop the Records with nulls in any of the columns.

# In[14]:


data.dropna(inplace=True)


# In[15]:


data.isnull().sum().sum()


# ### 4.1) Size column has sizes in Kb as well as Mb. To analyze, you’ll need to convert these to numeric.
# 
# ##### 4.1.1)Extract the numeric value from the column.

# In[16]:


data.Size


# In[17]:


data['Size'] = data.Size.str.replace('M', '')


# In[18]:


data.Size


# In[19]:


data.drop(data[data.Size.str.len() > 4].index, inplace=True)


# In[20]:


data.Size


# ##### 4.1.2) Multiply the value by 1,000, if size is mentioned in Mb.

# In[21]:


data['Size']=data[~data.Size.str.contains('k')]['Size'].astype(float) * 1000


# In[22]:


data.Size


# In[23]:


data.isnull().mean() 
# there is no need to drop null values in size as 30% null values in the column is the ground rule to drop.


# #### 4.2) Reviews is a numeric field that is loaded as a string field. Convert it to numeric (int/float).

# In[24]:


data['Reviews']=data.Reviews.astype(float)


# In[25]:


data.Reviews


# #### 4.3) Installs field is currently stored as string and has values like 1,000,000+. 
# 
# ##### 4.3.1) Treat 1,000,000+ as 1,000,000.
# ##### 4.3.2) remove ‘+’, ‘,’ from the field, convert it to integer.

# In[26]:


data['Installs']=data.Installs.str.replace('[,+]', '').astype(int)


# #### 4.4) Price field is a string and has ' Dollar symbol. Remove ‘Dollar’ sign, and convert it to numeric.

# In[27]:


data['Price']=data.Price.str.replace('$', '')


# In[28]:


data['Price']=data.Price.astype(float)


# In[29]:


data.Price


# #### 4.5) Sanity checks:
# 
# ##### 4.5.1) Average rating should be between 1 and 5 as only these values are allowed on the play store. Drop the rows
that have a value outside this range.

# In[30]:


data[(data['Rating'] > 1) & (data['Rating'] < 5)]
#Average rating which are between 1 and 5 as only these values are allowed on the play store.


# In[31]:


data[(data['Rating'] < 1) & (data['Rating'] > 5)]
# there are no values outside the range.So, there are no values to drop.


# ##### 4.5.2) Reviews should not be more than installs as only those who installed can review the app. If there are any such 
records, drop them.

# In[32]:


data[(data['Reviews']) > (data['Installs'])]


# In[33]:


data.drop(data[data['Reviews'] > data['Installs']].index,inplace=True)


# In[34]:


data[(data['Reviews']) > (data['Installs'])]
# now as we dropped there no are reviews which are more than installs.


# ##### 4.5.3) For free apps (Type = “Free”), the price should not be >0. Drop any such rows.

# In[35]:


data[(data.Type == 'Free') & (data.Price > 0)]
# no free apps has price greater than 0


# In[36]:


data


# ### 5) Performing univariate analysis: 
# - Boxplot for Price
# - Boxplot for Reviews
# - Histogram for Rating
# - Histogram for Size

# In[37]:


sns.set(rc={"figure.figsize":(12,5)})


# In[38]:


sns.boxplot(data['Price'])
plt.title('Boxplot for Price')
plt.ylabel('Price');


# In[39]:


sns.boxplot(data['Reviews'])
plt.title('Boxplot for Reviews')
plt.ylabel('Reviews');


# In[40]:


plt.hist(data.Rating);
plt.title('Histogram for Rating');


# In[41]:


plt.hist(data['Size']);
plt.title('Histogram for Size');


# ### 6) Outlier treatment: 
# 
# #### 6.1) Price: From the box plot, it seems like there are some apps with very high price. A price of 200 dollars for an 
application on the Play Store is very high and suspicious!
# ##### 6.1.1) Check out the records with very high price
# #### 6.2) Drop these as most seem to be junk apps

# In[42]:


# checking the price of the apps which are more than 200 dollars.
# we have already removed dollar sign.
data[data['Price'] > 200]


# In[43]:


data.drop(data[data['Price']>200].index,inplace=True)


# In[44]:


data[data['Price'] > 200] # now we dropped the junk apps which are more than 200 dollars.


# ### 6.2) Reviews: Very few apps have very high number of reviews. These are all star apps that don’t help with the analysis 
and, in fact, will skew it. Drop records having more than 2 million reviews.

# In[45]:


#to check the apps with > 2000000 reviews 
data[data['Reviews'] > 2000000]
# we can also use this command ....data[data['Reviews']>2000000]


# In[46]:


data.drop(data[data['Reviews']>2000000].index,inplace=True)
# we can try in this way also 
# data[data['Reviews']>2000000]
# data=data.dropna()


# In[47]:


data[data['Reviews'] > 2000000] 
# after dropping, now we cannot see the records having more than 2000000.


# ### 6.3) Installs:  There seems to be some outliers in this field too. Apps having very high number of installs should 
  be dropped from the analysis.
# 
# -Find out the different percentiles – 10, 25, 50, 70, 90, 95, 99
# 
# -Decide a threshold as cutoff for outlier and drop records having values more than that.

# In[48]:


data['Installs'].quantile([0.10, 0.25, 0.50, 0.70, 0.90, 0.95, 0.99])


# In[49]:


import numpy as np
q1 = np.percentile(data['Installs'], 25)
q3 = np.percentile(data['Installs'], 75)


# In[50]:


iqr=q3-q1
Threshold=1.5*iqr
q1,q3,iqr,Threshold


# In[51]:


# Here our goal is to find the threshold(i.e is high value) as cutoff for outliers.
# we can define the threshold as any value that falls more than 1.5 times the IQR i.e (1.5*IQR)
# below the first quartile or above the third quartile,and then remove any values that exceed the threshold 


# In[52]:


Threshold=q3+1.5*iqr
Threshold


# In[53]:


data.drop(data[data['Installs']>2485000.0].index,inplace=True)


# In[54]:


data.shape


# ## 7) Bivariate analysis: Let’s look at how the available predictors relate to the variable of interest, i.e., our 
target variable rating. Make scatter plots (for numeric features) and box plots (for character features) to assess the 
relations between rating and the other features.
# ### 7.1) Make scatter plot/joinplot for Rating vs. Price

# In[55]:


sns.scatterplot(x='Rating', y='Price',data=data);
plt.title('Scatterplot Rating vs Price');


# ### 7.1.1)What pattern do you observe? Does rating increase with price?
Observation1 : We can understand that Rating increased with Price by observing the above plot.
# ### 7.2) Make scatter plot/joinplot for Rating vs. Size

# In[56]:


sns.scatterplot(x='Rating', y='Size',data=data);
plt.title('Scatterplot Rating vs Size');


# ### 7.2.1) Are heavier apps rated better?
Observation2 : Yes,heavier apps rated better.
# ### 7.3) Make scatter plot/joinplot for Rating vs. Reviews

# In[57]:


sns.scatterplot(x='Rating', y='Reviews',data=data);
plt.title('Scatterplot Rating vs Reviews');


# ### 7.3.1) Does more review mean a better rating always?
Observation3 : Not necessarily,the number of reviews can provide an indication of the popularity and
               user engagement of an app, but it does not necessarily guarantee a better rating always.
               But in this case,by the above plot we understood that More reviews having better rating.
# ### 7.4) Make boxplot for Rating vs. Content Rating

# In[58]:


sns.boxplot(x='Rating', y='Content Rating',data=data);
plt.title('Boxplot for Rating vs. Content Rating');


# ### 7.4.1) Is there any difference in the ratings? Are some types liked better?
Observation4 : No, there is not much difference between the ratings of most apps, except for those
               rated as "adults only 18+," and those apps have higher ratings.
# ### 7.5) Make boxplot for Ratings vs. Category

# In[59]:


fig, ax = plt.subplots(figsize=(12,10))
sns.boxplot(x='Rating', y='Category',data=data, ax=ax);
plt.title('Boxplot for Ratings vs. Category');


# ### 7.5.1) Which genre has the best ratings?
Observation5 : By the above plot we can see that Events Category has Best ratings.
# ### 8) Data preprocessing
# - For the steps below, create a copy of the dataframe to make all the edits. Name it inp1.

# In[60]:


inp1 = data


# In[61]:


inp1.head()


# ### 8.1) Reviews and Install have some values that are still relatively very high. Before building a linear 
regression model, you need to reduce the skew. Apply log transformation (np.log1p) to Reviews and Installs.

# In[62]:


inp1.skew()


# In[63]:


inp1['Reviews'] = np.log1p(inp1['Reviews'])
inp1['Installs'] = np.log1p(inp1['Installs'])


# In[64]:


inp1.skew()


# ### 8.2) Drop columns App, Last Updated, Current Ver, and Android Ver. These variables are not useful for our task.

# In[65]:


inp1.drop(['App','Last Updated','Current Ver','Android Ver'],axis=1,inplace=True)


# In[66]:


inp1


# In[67]:


inp1.head()


# ### 8.3) Get dummy columns for Category, Genres, and Content Rating. This needs to be done as the models do not 
understand categorical data, and all data should be numeric. Dummy encoding is one way to convert character fields 
to numeric. Name of dataframe should be inp2.

# In[68]:


inp2 = inp1


# In[69]:


inp2


# In[80]:


import pandas as pd

# Create the dummy columns for Category, Genres, and Content Rating
category_dummies = pd.get_dummies(inp1['Category'], prefix='Category')
genres_dummies = pd.get_dummies(inp1['Genres'], prefix='Genres')
content_rating_dummies = pd.get_dummies(inp1['Content Rating'], prefix='Content_Rating')

# Concatenate the original dataframe with the dummy columns
inp2 = pd.concat([inp1, category_dummies, genres_dummies, content_rating_dummies], axis=1)


# In[81]:


inp2


# ### 9) Train test split  and apply 70-30 split. Name the new dataframes df_train and df_test.
# ### 10) Separate the dataframes into X_train, y_train, X_test, and y_test.

# In[82]:


from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import mean_squared_error as mse


# In[94]:


from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
df_train, df_test = train_test_split(inp2, test_size=0.3, random_state=42)

# Separate the training data into X_train (features) and y_train (target variable)
X_train = df_train.drop(['Rating'], axis=1)
y_train = df_train['Rating']

# Separate the testing data into X_test (features) and y_test (target variable)
X_test = df_test.drop(['Rating'], axis=1)
y_test = df_test['Rating']


# In[99]:


inp2.dtypes

