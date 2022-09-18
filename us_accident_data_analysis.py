#!/usr/bin/env python
# coding: utf-8

# # US Accidents Exploratory Data Analysis

# ##### Information about data: Source- Kaggle Contains Information about Accidents in USA Useful to prevent accidents

# ### Download Data

# In[8]:


import pandas as pd
import matplotlib.pyplot as plt


# In[11]:


import io
df = pd.read_csv('US_Accidents_Dec21_updated.csv')


# ### Data Preparation and Cleaning
# ##### --Load file using pandas -look at some information about the data & the column -fix any missing or incoreect values

# In[12]:


df.shape


# In[13]:


df.info()


# In[14]:


df.describe()


# In[15]:


numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

numeric_df = df.select_dtypes(include=numerics)
len(numeric_df.columns)


# In[16]:


miss_percent=df.isna().sum().sort_values(ascending=False)/len(df)
miss_percent[miss_percent != 0].plot(kind='barh')


# ##### Remove columns that you don't want to use

# In[17]:


df.drop('Precipitation(in)',axis=1,inplace=True)


# In[18]:


df.drop('Wind_Chill(F)',axis=1,inplace=True)


# In[19]:


df.drop('Number',axis=1,inplace=True)


# ## Exploratory Analysis and visualization

# ### Columns we will analyze

# #### 1 City
# #### 2 Start Time
# #### 3 Start Lat, Start Long
# #### 4 Temperature
# #### 5 Weather Condition

# ### Which 20 cities have most number of accidents?

# In[20]:


cities=df.City.unique()
len(cities)


# In[21]:


cities_by_accident=df.City.value_counts()
cities_by_accident.head(20)


# In[22]:


cities_by_accident[:20].plot(kind='barh')


# In[23]:


import seaborn as sns
sns.set_style("darkgrid")


# In[24]:


sns.distplot(cities_by_accident)


# In[25]:


sns.histplot(cities_by_accident,log_scale=True)


# In[26]:


cities_by_accident[cities_by_accident==1]


# In[27]:


high_accident_cities=cities_by_accident[cities_by_accident>=1000]
low_accident_cities=cities_by_accident[cities_by_accident<1000]


# In[28]:


len(high_accident_cities)/len(cities)


# In[29]:


sns.distplot(high_accident_cities)


# In[30]:


sns.distplot(low_accident_cities)


# #### Start Time

# In[31]:


df.Start_Time


# In[32]:


df.Start_Time=pd.to_datetime(df.Start_Time)


# In[33]:


sns.distplot(df.Start_Time.dt.hour,bins=24,kde=False, norm_hist=True)


# #####  - high perecentage of accidents occur between 3 PM -6 PM(probabaly because traffic is higher beacause people go to home from work.)
# ##### - Another time slot is 6 AM- 9 AM(probabaly because people leave for work in hurry)

# In[34]:


sns.distplot(df.Start_Time.dt.dayofweek,bins=7,kde=False, norm_hist=True)


# ##### - On weekdays, the peak of accidents is in the evening between 4 PM to 6 Pm and in the morning between 6 AM to 9 AM.

# #### Start Latitude and Longitude

# In[37]:


sns.scatterplot(x=df.Start_Lng,y=df.Start_Lat,size=0.00001)


# In[41]:


df.describe()


# ### Are there more accidents is warmer or colder areas?

# In[42]:


sns.jointplot(x=df['Temperature(F)'], y=df['Humidity(%)'])


# In[43]:


sns.lineplot(x=df['Temperature(F)'], y=df['Humidity(%)'])


# ##### From the above charts, We can say that there is relation between humidity and tempearture when accident occurs. Most of the accidents occurs when Humidity is high and Teperature is low. This can be Winter season when it is snowing. Other possible reason can be high temperature with low humidity.

# ### Which 5 States has highest number of accidents?

# In[44]:


states=df.State.unique()
states_by_accident=df.State.value_counts()


# In[45]:


top5_states=states_by_accident.head(5)
top5_states


# In[46]:


states_by_accident_per_capita=(df.State.value_counts()/len(df))*100
states_by_accident_per_capita.head(10)


# In[47]:


states_by_accident[:10].plot(kind='barh')


# ### Analysis for particulary 'New Jersey'

# In[48]:


NJ_count=df.loc[df['State']=='NJ']
len(NJ_count)


# In[49]:


NJ_cities=NJ_count.City.value_counts()
NJ_cities[:10].plot(kind='barh')


# ### Summary and Conclusion

# #### Insights

# ##### 1. New York has less than 5000 accidents per year inspite of being the most populated city in USA.
# ##### 2. There are more number of accidents around the coast.
# ##### 3. The number of accidents per city decreases exponentially.
# ##### 4. Less than 5% of cities have more than 1000 accidents.
# ##### 5. Over 1200 cities have reported just 1 accident. (needs to be investigated)
# ##### 6. Number of accidents are more when Humidity is high and Teperature is low and high temperature with low humidity.
# ##### 7. Newark city has highest number of accidents in New Jersey State.
