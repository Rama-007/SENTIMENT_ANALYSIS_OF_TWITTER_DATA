import os,sys,time,csv,random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn import svm
from sklearn import neural_network
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

fp=open('training.1600000.processed.noemoticon.csv','r')
reader=csv.reader(fp)
data=[]
for line in reader:
	temp_data=line[len(line)-1]
	label=int(line[0])
	words=temp_data.split()
	for i in range(0,len(words)):
		if("http" in words[i]):
			words[i]=''
		else:
			words[i]=words[i].lower()
	temp_data=' '.join(words)
	data.append((temp_data,label))
	
print('read complete')
random.shuffle(data)
data=data[:40000]
print('shuffle complete')
L=len(data)
train_data=[i[0] for i in data[:int(0.8*L)]]
train_labels=[i[1] for i in data[:int(0.8*L)]]
test_data=[i[0] for i in data[int(0.8*L):]]
test_labels=[i[1] for i in data[int(0.8*L):]]

tf=TfidfVectorizer(min_df=5,max_df = 0.8,sublinear_tf=True,use_idf=True,decode_error='ignore')
# cv=CountVectorizer()


train_tf_vectors=tf.fit_transform(train_data)
test_tf_vectors=tf.transform(test_data)

print('TF-IDF complete')


# train_cv_vectors=cv.fit_transform(train_data)
# test_cv_vectors=cv.transform(test_data)

classifier_lin=neural_network.MLPClassifier()
#classifier_lin = svm.SVC(kernel='linear')
classifier_lin.fit(train_tf_vectors,train_labels)
print("training completed")
prediction_rbf=classifier_lin.predict(test_tf_vectors)
print("TF-IDF")
print(classification_report(test_labels, prediction_rbf))
print(accuracy_score(test_labels, prediction_rbf))

"""
print("BNB")
classifier_bnb=BernoulliNB()
classifier_bnb.fit(train_tf_vectors,train_labels)
print("training completed")
prediction_bnb=classifier_bnb.predict(test_tf_vectors)
print(classification_report(test_labels, prediction_bnb))
print(accuracy_score(test_labels, prediction_bnb))
"""
