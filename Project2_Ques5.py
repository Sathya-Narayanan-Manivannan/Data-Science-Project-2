
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

#3.5 Deep Learning
#Reference - Dr. Gene Moo Lee's notes for Data Science
#Reference - http://stackoverflow.com/questions/12572362/get-a-string-after-a-specific-substring

#Question 1
'Collecting URL of all images from android and ios'
image_urls=set()
from bs4 import BeautifulSoup

for filename in ios_file_list:
      webpageopen_ios = open(filename, 'r').read()
      webpage_ios = BeautifulSoup(webpageopen_ios, 'lxml')
      ios_screenshots = webpage_ios.find_all('img', {'itemprop' : 'screenshot'})
      for tags in ios_screenshots:
          url=str(tags).partition(' ')[2].split()[5].split('"')[1]
          image_urls.add(url)



for filenameA in android_file_list:
    webpageopen_android = open(filenameA, 'r').read()
    webpage_android = BeautifulSoup(webpageopen_android, 'lxml')
    android_screenshots = webpage_android.find_all('img', {'class' : 'full-screenshot'})
    for tag in android_screenshots:
        urla=str(tag).partition(' ')[2].split()[9].split('"')[1]
        image_urls.add(urla)


#Taking a backup of the images
#import pickle
#with open("image_url.pickle", "wb") as output:
    #pickle.dump(image_urls, output)


#Question 2
#Reference - http://stackoverflow.com/questions/8286352/how-to-save-an-image-locally-using-python-whose-url-address-i-already-know
#Reference - http://stackoverflow.com/questions/7391945/how-do-i-read-image-data-from-a-url-in-python

'Downloading all images'
image_urls_lst=list(image_urls)

import urllib
from PIL import Image
from StringIO import StringIO

for i in range(len(image_urls_lst)):
        
    if image_urls_lst[i].startswith('http'):
        opening_img = urllib.urlopen(image_urls_lst[i]).read()
        output_img=open('imgfile'+str(i)+'.jpg','wb')
        output_img.write(opening_img)
        output_img.close()
     
    else:
        opening_img = urllib.urlopen('http:'+image_urls_lst[i]).read()
        Img=Image.open(StringIO(opening_img))
        Img.save('imgfile'+str(i)+'.jpg', "JPEG")


#Question 3:
'Extracting tags with probabilities using Tensor Flow'
#Installed Python 3.5 and installed Tensor Flow package
#import pip
#pip.main(['install','tensorflow'])
'Downloaded image_classify_by_tensorflow.py from Dr. Gene Moo Lee notes for Data Science'
'From command line terminal, gave the following command'
'python image_classify_by_tensorflow.py --image_file=imgfile21.jpg'
'got an output file imgfile21.jpg_tf_analyzed.tsv'

