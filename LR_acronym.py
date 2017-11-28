import os,sys,time,csv,random,re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn import svm
from sklearn import neural_network
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn import ensemble
from sklearn import linear_model

acronyms={'afk':'away from keyboard',
'brb':'be right back',
'bbiab':'be back in a bit',
'bbl':'be back later',
'ttfn':'ta ta for now',
'bbs':'be back soon',
'btw':'by the way',
'hagn':'have a good night',
'kiss':'keep it simple stupid',
'kit':'keep in touch',
'eg':'evil grin',
'beg':'big evil grin',
'nyob':'none of your business',
'omg':'oh my god',
'pm':'private message',
'pos':'parents over shoulder',
'ttyl':'talk to you later',
'ltns':'long time no see',
'ssdd':'same shit different day',
'idk':'i dont know',
'lol':'laughing out loud',
'lylab':'love you like a brother',
'lylas':'love you like a sister',
'gmta':'great minds think alike',
'rofl':'rolling on floor laughing',
'lmao':'laughing my ass off',
'ty':'thank you',
'tyvm':'thank you very much',
'yw':'your welcome',
'np':'no problem',
'wtg':'way to go',
'phat':'pretty, hot, and tempting',
'bf':'boyfriend',
'gf':'girlfriend',
'gj':'good job',
'swak':'sealed with a kiss',
'bf':'best friend',
'bff':'best friends forever',
'bffl':'best friends for life',
'gr8':'great',
}

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

classifier_lin=linear_model.LogisticRegressionCV()
classifier_lin.fit(train_tf_vectors,train_labels)
print("training completed")
prediction_rbf=classifier_lin.predict(test_tf_vectors)
print("TF-IDF")
print(classification_report(test_labels, prediction_rbf))
print(accuracy_score(test_labels, prediction_rbf))



for i in range(0,len(data)):
	temp=data[i][0].split(' ')
	for j in range(0,len(temp)):
		if temp[j] in acronyms:
			temp[j]=acronyms[temp[j]]
	temp1=' '.join(temp)
	data[i]=(temp1,data[i][1])
			

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

classifier_lin=linear_model.LogisticRegressionCV()
classifier_lin.fit(train_tf_vectors,train_labels)
print("training completed")
prediction_rbf=classifier_lin.predict(test_tf_vectors)
print("TF-IDF")
print(classification_report(test_labels, prediction_rbf))
print(accuracy_score(test_labels, prediction_rbf))

"""
for i in range(0,len(data)):
	data[i]=(re.sub('#.*? ','',data[i][0]),data[i][1])
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

classifier_lin=linear_model.LogisticRegressionCV()
classifier_lin.fit(train_tf_vectors,train_labels)
print("training completed")
prediction_rbf=classifier_lin.predict(test_tf_vectors)
print("TF-IDF")
print(classification_report(test_labels, prediction_rbf))
print(accuracy_score(test_labels, prediction_rbf))
"""
"""
print("BNB")
classifier_bnb=BernoulliNB()
classifier_bnb.fit(train_tf_vectors,train_labels)
print("training completed")
prediction_bnb=classifier_bnb.predict(test_tf_vectors)
print(classification_report(test_labels, prediction_bnb))
print(accuracy_score(test_labels, prediction_bnb))
"""
