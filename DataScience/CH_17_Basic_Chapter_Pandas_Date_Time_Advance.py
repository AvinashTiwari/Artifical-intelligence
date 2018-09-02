
# coding: utf-8

# In[1]:


import pandas as pd
import datetime as dt


mday = dt.date(2017,9,2)


# In[2]:


mday


# In[3]:


mday.year


# In[4]:


mday.day


# In[5]:


mday.month


# In[6]:


mydayandtime = dt.datetime(2017,9,2, 15,24,30)


# In[7]:


mydayandtime.hour


# In[8]:


mydayandtime.minute


# In[9]:


mydayandtime.second


# In[10]:


pd.Timestamp('2017-09-02')


# In[11]:


pd.Timestamp('2017/09/02')


# In[12]:


pd.Timestamp('2017,09,02')


# In[13]:


pd.Timestamp('2017,09,02 15:29:10')


# In[14]:


date = ['2017/08/31', '2017/09/01', '2017/09/02']
pd.DatetimeIndex(date)


# In[15]:


date2 = [dt.date(2017,8,31), dt.date(2017,8,30),dt.date(2017,8,29)]
pd.DatetimeIndex(date2)


# In[16]:


food = ['Pizza', 'Burgger', 'Salad']
pd.Series  (data = food, index=date)


# In[17]:


dirtyyimes = pd.Series(['2017-09-02', '2017/09/01', '2017', 'July  4 ,2017','Avinash'])
dirtyyimes


# In[18]:


pd.to_datetime(dirtyyimes, errors='coerce')


# In[19]:


pd.Period('2017-09-02', freq='21D')


# In[20]:


pd.Period('2017-09-02', 'W-SUN')


# In[21]:


pd.Period('2017-09-02', freq='W-TUE')


# In[22]:


myrange = pd.date_range(start='2017-09-02', end='2017-12-25', freq='D')


# In[23]:


myrange


# In[24]:


myrange2 = pd.date_range(start='2017-09-02', end='2037-12-25', freq='A')
myrange2


# In[25]:


myrange3 = pd.date_range(start='2017-09-02', periods = 140, freq='D')
myrange3


# In[26]:


myrange4 = pd.date_range(start='2017-12-24', periods = 140, freq='12h')
myrange4


# In[27]:


myseries= pd.Series(myrange4)
myseries


# In[28]:


myseries.dt.is_month_end


# In[29]:


myseries[myseries.dt.is_month_end]


# In[31]:


myseries[myseries.dt.is_quarter_end]


# In[33]:


ts = pd.Timestamp('2017-09-03 10:27:30')
ts


# In[34]:


ts.weekday_name


# In[36]:


ts.is_month_end


# In[37]:


ts.is_month_start


# In[38]:


myrange5 = pd.date_range(start='2017-12-24', periods = 140, freq='12h')
myrange5

