#Reference - Dr. Gene Moo Lee's notes for Data Science
#Reference - http://stackoverflow.com/questions/12591575/python-typeerror-must-be-encoded-string-without-null-bytes-not-str
#Reference - http://stackoverflow.com/questions/2909975/python-list-directory-subdirectory-and-files

#3.1 Web Scraping
#Question 1:

'To get the list of ios and android files'

import os
from fnmatch import fnmatch

root = 'data'
pattern_ios = "*_ios.html"
pattern_android= "*_android.html"
ios_file_list=[]
android_file_list=[]

for path, subdirs, files in os.walk(root):
    for name in files:
        if fnmatch(name, pattern_ios):
            file_path_ios = os.path.join(path, name)
            ios_file_list.append(file_path_ios)
        elif fnmatch(name, pattern_android):
            file_path_android = os.path.join(path, name)
            android_file_list.append(file_path_android)

'To create a json file with ios file names and corresponding Current version rating, All version rating and File size'

from bs4 import BeautifulSoup
import string
counts_size_ios = dict()

for filename in ios_file_list:
      webpageopen = open(filename, 'r').read()
      webpage = BeautifulSoup(webpageopen, 'lxml')
      count_ver = webpage.find_all('span', {'class' : 'rating-count'})
      try:
         ios_current_ratings = str(count_ver[0].get_text()).split()[0]
      except:
         ios_current_ratings = 'NA'
      try:
          ios_all_ratings = str(count_ver[1].get_text()).split()[0]
      except:
          ios_all_ratings = 'NA'
      #print ios_current_ratings, ios_all_ratings

      all = webpage.find_all('ul', {'class' : 'list'})
      to_find_size=all[0].get_text()
      norm=filter(lambda x: x in string.printable, to_find_size)
      try:
          ios_file_size = norm.split()[6].encode('utf-8')
      except:
          ios_file_size = 'NA'
      #print ios_file_size
      counts_size_ios[filename] = ios_current_ratings + ',' + ios_all_ratings + ',' + ios_file_size
 
import json
with open('counts_size_ios.json', 'w') as f:
    json.dump(counts_size_ios, f, indent=4)


'To create a json file with android file names and corresponding ratings and File size' 

from bs4 import BeautifulSoup
counts_size_android = dict()

for filename_android in android_file_list:
    webpage_android_read = open(filename_android, 'r').read()
    webpage_android = BeautifulSoup(webpage_android_read, 'lxml')
    
    try:
        avg_count_android = webpage_android.find('div', {'class' : 'score'})    
        android_avg_rating = str(avg_count_android).split()[3]
        #print android_avg_rating
    except:
        android_avg_rating = 'NA'
   
    try:
        total_count_android = webpage_android.find('span', {'class' : 'reviews-num'})
        android_total_ratings = ''.join(str(total_count_android).split()[2].split(','))
        #print android_total_ratings
    except:
        android_total_ratings = 'NA'
    
    android_ratings = webpage_android.find_all('span', {'class' : 'bar-number'})
    
    try:
        android_ratings_1 = ''.join(str(android_ratings[4]).split()[2].split(','))
        #print android_ratings_1
    except:
        android_ratings_1 = 'NA'
    
    try:
        android_ratings_2 = ''.join(str(android_ratings[3]).split()[2].split(','))
        #print android_ratings_2
    except:
        android_ratings_2 = 'NA'
    
    try:
        android_ratings_3 = ''.join(str(android_ratings[2]).split()[2].split(','))
        #print android_ratings_3
    except:
        android_ratings_3 = 'NA'
    
    try:
        android_ratings_4= ''.join(str(android_ratings[1]).split()[2].split(','))
        #print android_ratings_4
    except:
        android_ratings_4 = 'NA'
    
    try:
        android_ratings_5 = ''.join(str(android_ratings[0]).split()[2].split(','))
        #print android_ratings_5
    except:
        android_ratings_5 = 'NA'
    
    android_size = webpage_android.find('div', {'itemprop' : 'fileSize'})
    try:
        android_file_size = str(android_size).split()[-2][:-1]
        #print android_file_size
    except:
        android_file_size = 'NA'
    
    counts_size_android[filename_android] = android_avg_rating + ',' + android_total_ratings + ',' + android_ratings_1 + ',' + android_ratings_2 + ',' + android_ratings_3 + ',' + android_ratings_4 + ',' + android_ratings_5 + ',' + android_file_size


import json
with open('counts_size_android.json', 'w') as f:
    json.dump(counts_size_android, f, indent=4)
