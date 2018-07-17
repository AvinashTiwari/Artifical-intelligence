
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


#"2018_Weather_Data.csv",
def read_csv_data(file_path, start_row, run_row):
    data_frame = pd.read_csv(  file_path,
                 sep=",",
                 names=['Max temp','Min temp', 'Mean temp', 'Total  precip'],
                 usecols=[5,7,9,19],
                 skiprows=start_row,
                 nrows=run_row)
    return data_frame


# In[3]:


print(read_csv_data("2018_Weather_Data.csv", 1 ,195))


# In[4]:


data_frame = read_csv_data("2018_Weather_Data.csv", 1 ,195)
data_frame['Max temp']


# In[5]:


def calculate_labels(list_of_temps):
    list_of_labels = []
    for i in range(len(list_of_temps) - 1):
        if list_of_temps[i + 1] >= list_of_temps[i]:
            list_of_labels.append([1, 0])
        else:
            list_of_labels.append([0, 1])
    return list_of_labels


# In[6]:


def calculate_differences(list_of_factors):
    list_of_differences = []
    for i in range(len(list_of_factors) - 1):
        if list_of_factors[i + 1] >= list_of_factors[i]:
            list_of_differences.append(1)
        else:
            list_of_differences.append(0)
    return list_of_differences


# In[7]:


def build_data_subset(file_path, start_row, num_rows):
    data_frame = read_csv_data(file_path, start_row, num_rows)
    max_temp = calculate_differences(data_frame['Max temp'])
    min_temp = calculate_differences(data_frame['Min temp'])
    mean_temp = calculate_differences(data_frame['Mean temp'])
    precip = calculate_differences(data_frame['Total  precip'])
    labels = calculate_labels(data_frame['Mean temp'])

    formatted_data = []
    for i in range(len(max_temp)):
        data_point = [max_temp[i], min_temp[i], mean_temp[i], precip[i]]
        formatted_data.append(data_point)

    return formatted_data, labels


# In[8]:


data ,labels = build_data_subset("2018_Weather_Data.csv", 1 ,15)
print(data)
print(labels)

