#3.2 Data Organization
#Reference - Dr. Gene Moo Lee's notes for Data Science
#Reference - http://pandas.pydata.org/pandas-docs/stable/missing_data.html

#Question 1:
'To create a dictionary, where the key is datetime object and the value is a dictionary with extracted values from iOS and Android HTML files'    

from datetime import datetime
import json

with open('C:/Users/sande/Downloads/python project 2/counts_size_android.json', 'r') as a:
    android_time_stamps = json.load(a)
with open('C:/Users/sande/Downloads/python project 2/counts_size_ios.json','r') as i:
    ios_time_stamps = json.load(i)

import numpy as np
date_time_count_dict = dict()

for i in sorted(android_time_stamps.keys()):   
    for j in sorted (ios_time_stamps.keys()):
        if i[:22] == j[:22]:
            value_dict = dict()
            if str(ios_time_stamps[j].split(',')[0]) == 'NA':
                value_dict['ios_current_ratings'] = np.nan
            else:
                value_dict['ios_current_ratings'] = int(ios_time_stamps[j].split(',')[0])
            if str(ios_time_stamps[j].split(',')[1]) =='NA':
                value_dict['ios_all_ratings'] = np.nan
            else:
                value_dict['ios_all_ratings'] = int(ios_time_stamps[j].split(',')[1])
            if str(ios_time_stamps[j].split(',')[2]) =='NA':
                value_dict['ios_file_size'] = np.nan
            else:
                value_dict['ios_file_size'] = int(ios_time_stamps[j].split(',')[2])
            if str(android_time_stamps[i].split(',')[0]) == 'NA':
                value_dict['android_avg_rating'] = np.nan
            else:
                value_dict['android_avg_rating'] = float(android_time_stamps[i].split(',')[0])
            if str(android_time_stamps[i].split(',')[1]) =='NA':
                value_dict['android_total_ratings'] = np.nan
            else:
                value_dict['android_total_ratings'] = int(android_time_stamps[i].split(',')[1])
            if str(android_time_stamps[i].split(',')[2]) =='NA':
                value_dict['android_rating_1'] = np.nan
            else:
                value_dict['android_rating_1'] = int(android_time_stamps[i].split(',')[2])
            if str(android_time_stamps[i].split(',')[3]) == 'NA':
                value_dict['android_rating_2'] = np.nan
            else:
                value_dict['android_rating_2'] = int(android_time_stamps[i].split(',')[3])
            if str(android_time_stamps[i].split(',')[4]) =='NA':
                value_dict['android_rating_3'] = np.nan
            else:
                value_dict['android_rating_3'] = int(android_time_stamps[i].split(',')[4])
            if str(android_time_stamps[i].split(',')[5]) == 'NA':
                value_dict['android_rating_4'] = np.nan
            else:
                value_dict['android_rating_4'] = int(android_time_stamps[i].split(',')[5])
            if str(android_time_stamps[i].split(',')[6]) =='NA':
                value_dict['android_rating_5'] = np.nan
            else:
                value_dict['android_rating_5'] = int(android_time_stamps[i].split(',')[6])
            if str(android_time_stamps[i].split(',')[7]) == 'NA':
                value_dict['android_file_size'] = np.nan
            else:
                value_dict['android_file_size'] = int(android_time_stamps[i].split(',')[7])           
            
            common_date = i.split('\\')[1].split('-')
            common_time = i.split('\\')[2].split('_')
            date_time_count_dict[datetime(int(common_date[0]),int(common_date[1]),int(common_date[2]),int(common_time[0]),int(common_time[1]))] =  value_dict
            break
        else:
            continue
         
'''
#Took a backup of the dictionary for verification
with open('date_time_count_dict.json', 'w') as d:
    json.dump(date_time_count_dict, d, indent=4)
'''

#Question 2:
'Converting date_time_count_dict dictionary into a Pandas dataframe'

import pandas as pd

df = pd.DataFrame(date_time_count_dict.values(),index= date_time_count_dict.keys(), columns=date_time_count_dict.values()[0].keys())
#print df

#Question 3:
'Saving the dataframe into three formats (JSON, CSV, Excel)'


'JSON'
df.to_json('C:/Users/sande/Downloads/python project 2/data.json')

'CSV'
df.to_csv('C:/Users/sande/Downloads/python project 2/data.csv')

'Excel'
#df.to_excel('data.xlsx', sheet_name='Sheet1')
#Reference - http://stackoverflow.com/questions/29459461/pandas-dataframe-to-excel-sheet
from pandas import ExcelWriter

writer = ExcelWriter('C:/Users/sande/Downloads/python project 2/data.xlsx')
df.to_excel(writer,'Sheet1')
writer.save()

#3.3 Data Exploration
#Reference - Dr. Gene Moo Lee's notes for Data Science
#Reference - https://chrisalbon.com/python/pandas_missing_data.html
#Reference - http://stackoverflow.com/questions/17764619/pandas-dataframe-group-year-index-by-decade

'To handle the missing values, we have planned to replace those with mean of that particular date'

df['ios_current_ratings'].fillna(df.groupby(df.index.date)['ios_current_ratings'].mean(), inplace=True)
df['ios_all_ratings'].fillna(df.groupby(df.index.date)['ios_all_ratings'].mean(), inplace=True)
df['ios_file_size'].fillna(df.groupby(df.index.date)['ios_file_size'].mean(), inplace=True)
df['android_avg_rating'].fillna(df.groupby(df.index.date)['android_avg_rating'].mean(), inplace=True)
df['android_total_ratings'].fillna(df.groupby(df.index.date)['android_total_ratings'].mean(), inplace=True)
df['android_rating_1'].fillna(df.groupby(df.index.date)['android_rating_1'].mean(), inplace=True)
df['android_rating_2'].fillna(df.groupby(df.index.date)['android_rating_2'].mean(), inplace=True)
df['android_rating_3'].fillna(df.groupby(df.index.date)['android_rating_3'].mean(), inplace=True)
df['android_rating_4'].fillna(df.groupby(df.index.date)['android_rating_4'].mean(), inplace=True)
df['android_rating_5'].fillna(df.groupby(df.index.date)['android_rating_5'].mean(), inplace=True)
df['android_file_size'].fillna(df.groupby(df.index.date)['android_file_size'].mean(), inplace=True)
#df['android_file_size'].fillna(df['android_file_size'].mean(), inplace=True)
#df['android_file_size'].fillna(df.groupby(df.index.date)['android_file_size'].mean(), inplace=True)

'Even after replacing missing values with the mean of their respective dates, we found that files from few dates did not provide any count'
'We are dropping the above mentioned files from a date in which either of the count did not have any value' 
#Reference - http://stackoverflow.com/questions/13413590/how-to-drop-rows-of-pandas-dataframe-whose-value-in-certain-columns-is-nan


cleaned_df = df.dropna()
cleaned_df=cleaned_df.sort_index()
writer1 = ExcelWriter('C:/Users/sande/Downloads/python project 2/cleaned_data.xlsx')
cleaned_df.to_excel(writer1,'Sheet1')
writer1.save()

#Question 1:
'describe() method to find the count/mean/std/min/25%/50%75%/max'
#print cleaned_df.describe()

#Question 2:
'''scatter_matrix() method to find pairs of variables with high correlations'''

'''
#Tried Seaborn for verification
import pip
pip.main(['install','seaborn'])
import matplotlib.pyplot as plt
import seaborn as sb
sb.pairplot(cleaned_df)
plt.show()

'''
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix

scatter_matrix(cleaned_df, figsize=(25,25))
plt.show()


#Question 3:

'Pearson correlation coefficients'
import numpy

print numpy.corrcoef(cleaned_df.android_rating_1, cleaned_df.android_rating_5)[0][1]

print numpy.corrcoef(cleaned_df.android_rating_1, cleaned_df.android_total_ratings)[0][1]

print numpy.corrcoef(cleaned_df.android_rating_3, cleaned_df.android_rating_1)[0][1]

print numpy.corrcoef(cleaned_df.android_rating_3, cleaned_df.android_rating_5)[0][1]

print numpy.corrcoef(cleaned_df.android_rating_2, cleaned_df.android_rating_1)[0][1]

print numpy.corrcoef(cleaned_df.android_rating_2, cleaned_df.android_rating_3)[0][1]

print numpy.corrcoef(cleaned_df.android_rating_2, cleaned_df.android_rating_5)[0][1]

print numpy.corrcoef(cleaned_df.android_rating_4, cleaned_df.android_rating_1)[0][1]

print numpy.corrcoef(cleaned_df.android_rating_4, cleaned_df.android_rating_2)[0][1]

print numpy.corrcoef(cleaned_df.android_rating_4, cleaned_df.android_rating_3)[0][1]

print numpy.corrcoef(cleaned_df.android_rating_5, cleaned_df.android_rating_4)[0][1]

print numpy.corrcoef(cleaned_df.android_rating_3, cleaned_df.android_total_ratings)[0][1]

print numpy.corrcoef(cleaned_df.android_rating_2, cleaned_df.android_total_ratings)[0][1]

print numpy.corrcoef(cleaned_df.android_rating_5, cleaned_df.android_total_ratings)[0][1]

print numpy.corrcoef(cleaned_df.android_rating_4, cleaned_df.android_total_ratings)[0][1]

print numpy.corrcoef(cleaned_df.android_rating_1, cleaned_df.ios_all_ratings)[0][1]


#Question 4:
#Reference - https://blog.mafr.de/2012/03/11/time-series-data-with-matplotlib/
#import matplotlib.dates as mdates
plt.figure(figsize=(15,15))
plt.plot_date(cleaned_df.index,cleaned_df.android_total_ratings)
plt.plot_date(cleaned_df.index,cleaned_df.ios_all_ratings)
plt.plot_date(cleaned_df.index,cleaned_df.android_rating_5)
plt.plot_date(cleaned_df.index,cleaned_df.android_rating_4)
plt.plot_date(cleaned_df.index,cleaned_df.android_rating_3)
plt.plot_date(cleaned_df.index,cleaned_df.android_rating_2)
plt.plot_date(cleaned_df.index,cleaned_df.android_rating_1)
plt.show()

plt.figure(figsize=(15,15))
plt.plot_date(cleaned_df.index,cleaned_df.android_avg_rating)
plt.show()

plt.figure(figsize=(15,15))
plt.plot_date(cleaned_df.index,cleaned_df.ios_current_ratings)
plt.show()

plt.figure(figsize=(15,15))
plt.plot_date(cleaned_df.index,cleaned_df.ios_file_size)
plt.plot_date(cleaned_df.index,cleaned_df.android_file_size)
plt.show()


'''
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates

times = pd.date_range('2016-07-21', periods=max(cleaned_df.android_total_ratings), freq='10min')

fig, ax = plt.subplots(1)
fig.autofmt_xdate()
plt.plot(times, range(times.size))

xfmt = mdates.DateFormatter('%d-%m-%y %H:%M')
ax.xaxis.set_major_formatter(xfmt)

plt.show()

'''
#xls_file = pd.ExcelFile('cleaned_data.xlsx')
#newdf = xls_file.parse('Sheet1')
'3.4 Prediction Model'
'Reference : Dr. Gene Moo Lee Notes'
'Building a linear regression model using cross validation and predict for future date'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import datetime,time
'''
xls_file = pd.ExcelFile('C:/Users/sande/Downloads/python project 2/cleaned_data.xlsx')
cleaned_df = xls_file.parse('Sheet1')
#withindexdf=cleaned_df.reset_index()
'''
x=cleaned_df.drop(['android_total_ratings','ios_file_size','ios_current_ratings','android_file_size','ios_all_ratings','android_rating_5','android_rating_3','android_rating_1'],1)
y=cleaned_df['android_total_ratings']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)
#4352574,856213,285115
#x_train=x[0:8355]
#y_train=y[0:8355]
#x_test=x[8355:]
#y_test=y[8355:]
clf=LinearRegression()
model=clf.fit(x_train,y_train)
#x_new=(['1478062200','4.1','631647','216143'])
y_pred= model.predict(x_test)
print y_pred
newx_test=np.array([4,856213,285115]).reshape(1,-1)
newy_pred=model.predict(newx_test)
print("Predicted Value of Android Total Rating for 11/01/2016 23:50:00 is {}").format(newy_pred)


#print np.sqrt(metrics.mean_squared_error(y_test,y_pred))

#7.1870696627e-10
'correlation coefficients of ios_all_ratings' 
'''
android_total_ratings    0.960481
ios_file_size            0.722526
ios_current_ratings     -0.467778
android_rating_5         0.961306
android_avg_rating       0.790789
android_file_size        0.654830
android_rating_4         0.959777
ios_all_ratings          1.000000
android_rating_2         0.966103
android_rating_3         0.960589
android_rating_1         0.949642
Name: ios_all_ratings, dtype: float64
'''
#x1=cleaned_df.drop(['ios_all_ratings','ios_current_ratings','android_file_size'],1)
x1=cleaned_df.drop(['ios_all_ratings','android_file_size','android_rating_5','android_rating_4','android_rating_2','android_rating_1','android_rating_3'],1)
y1=cleaned_df['ios_all_ratings']

x_train1,x_test1,y_train1,y_test1=train_test_split(x1,y1,test_size=0.2,random_state=1)
clf1=LinearRegression()
model1=clf1.fit(x_train1,y_train1)
y_pred1= model1.predict(x_test1)
print y_pred1
newx_test1=np.array([7005220,259,2436,4]).reshape(1,-1)
newy_pred1=model1.predict(newx_test1)
print("Predicted Value of IOS Total Rating for 11/01/2016 23:50:00 is {}").format(newy_pred1)
#print newy_pred1