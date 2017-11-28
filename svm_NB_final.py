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
from collections import defaultdict
from get_smiley import happy , sad
from linguistic_features import feature_getter,pos_getter, num_stop_words ,num_punc


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

sl=defaultdict(int)
for i in open("list_swear",'r').readlines():
	i=i.strip().lower()
	sl[i]=1

pos_e=defaultdict(int)
for i in open("postive_emotion",'r').readlines():
	i=i.strip().lower()
	pos_e[i]=1

neg_e=defaultdict(int)
for i in open("negative_emotion",'r').readlines():
	i=i.strip().lower()
	neg_e[i]=1

def pos_emo(text):
	words=text.split(" ")
	sc=0
	for word in words:
		if word.lower() in pos_e:
			sc+=1
	return sc

def neg_emo(text):
	words=text.split(" ")
	sc=0
	for word in words:
		if word.lower() in neg_e:
			sc+=1
	return sc


def swear_number(text):
	words=text.split(" ")
	sc=0
	for word in words:
		if word.lower() in sl:
			sc=+1
	return sc


def get_features(tweet):
	features=[]
	features.append(len(str(tweet)))
	features.append(len(str(tweet).split(" ")))
	features.append(len(set(str(tweet))))
	features.append(happy(str(tweet)))
	features.append(sad(str(tweet)))
	features.append(swear_number(str(tweet)))
	features.append(pos_emo(str(tweet)))
	features.append(neg_emo(str(tweet)))
	features=features+feature_getter(str(tweet))
	features=features+pos_getter(str(tweet))
	features.append(num_stop_words(str(tweet)))
	features.append(num_punc(str(tweet)))
	return features



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
data=data[:5000]

for i in range(0,len(data)):
	temp=data[i][0].split(' ')
	for j in range(0,len(temp)):
		if temp[j] in acronyms:
			temp[j]=acronyms[temp[j]]
	temp1=' '.join(temp)
	data[i]=(temp1,data[i][1])

for i in range(0,len(data)):
	data[i]=(re.sub('#.*? ','',data[i][0]),data[i][1])

print('shuffle complete')
L=len(data)
train_data=[i[0] for i in data[:int(0.8*L)]]
train_labels=[i[1] for i in data[:int(0.8*L)]]
test_data=[i[0] for i in data[int(0.8*L):]]
test_labels=[i[1] for i in data[int(0.8*L):]]

train_features=[]
for k in train_data:
	i=k
	i=re.sub(r'[^\x00-\x7F]+',' ', i)
	train_features.append(np.asarray(get_features(i)))
test_features=[]
for k in test_data:
	i=k
	i=re.sub(r'[^\x00-\x7F]+',' ', i)
	test_features.append(np.asarray(get_features(i)))

print('feature trained')


tf=TfidfVectorizer(min_df=5,max_df = 0.8,sublinear_tf=True,use_idf=True,decode_error='ignore')
# cv=CountVectorizer()


train_tf_vectors=tf.fit_transform(train_data)
test_tf_vectors=tf.transform(test_data)


# train_cv_vectors=cv.fit_transform(train_data)
# test_cv_vectors=cv.transform(test_data)

classifier_lin = svm.SVC(kernel='linear')
classifier_lin.fit(train_tf_vectors,train_labels)
print("training completed")
prediction_rbf=classifier_lin.predict(test_tf_vectors)
print("TF-IDF")
print(classification_report(test_labels, prediction_rbf))
print(accuracy_score(test_labels, prediction_rbf))

print("BNB")
classifier_bnb=BernoulliNB()
classifier_bnb.fit(train_tf_vectors,train_labels)
print("training completed")
prediction_bnb=classifier_bnb.predict(test_tf_vectors)
print(classification_report(test_labels, prediction_bnb))
print(accuracy_score(test_labels, prediction_bnb))



print '#############bi'
tf=TfidfVectorizer(min_df=5,max_df = 0.8,sublinear_tf=True,use_idf=True,decode_error='ignore',ngram_range=(1, 2))
# cv=CountVectorizer()


train_tf_vectors2=tf.fit_transform(train_data)
test_tf_vectors2=tf.transform(test_data)

print('TF-IDF complete')


# train_cv_vectors=cv.fit_transform(train_data)
# test_cv_vectors=cv.transform(test_data)

classifier_lin = svm.SVC(kernel='linear')
classifier_lin.fit(train_tf_vectors2,train_labels)
print("training completed")
prediction_rbf=classifier_lin.predict(test_tf_vectors2)
print("TF-IDF")
print(classification_report(test_labels, prediction_rbf))
print(accuracy_score(test_labels, prediction_rbf))

print("BNB")
classifier_bnb=BernoulliNB()
classifier_bnb.fit(train_tf_vectors,train_labels)
print("training completed")
prediction_bnb=classifier_bnb.predict(test_tf_vectors)
print(classification_report(test_labels, prediction_bnb))
print(accuracy_score(test_labels, prediction_bnb))


print '#############Uni+bi'
#final_train1=np.hstack([train_tf_vectors.toarray(),train_tf_vectors2.toarray()])
#final_test1=np.hstack([train_tf_vectors.toarray(),test_tf_vectors2.toarray()])

final_train1=np.concatenate((train_tf_vectors.toarray(),train_tf_vectors2.toarray()),axis=1)

final_test1=np.concatenate((test_tf_vectors.toarray(),test_tf_vectors2.toarray()),axis=1)

print('TF-IDF complete')


# train_cv_vectors=cv.fit_transform(train_data)
# test_cv_vectors=cv.transform(test_data)

classifier_lin = svm.SVC(kernel='linear')
classifier_lin.fit(final_train1,train_labels)
print("training completed")
prediction_rbf=classifier_lin.predict(final_test1)
print("TF-IDF")
print(classification_report(test_labels, prediction_rbf))
print(accuracy_score(test_labels, prediction_rbf))

print("BNB")
classifier_bnb=BernoulliNB()
classifier_bnb.fit(final_train1,train_labels)
print("training completed")
prediction_bnb=classifier_bnb.predict(final_test1)
print(classification_report(test_labels, prediction_bnb))
print(accuracy_score(test_labels, prediction_bnb))


print '#############features'
final_train=np.hstack([train_features,final_train1])
final_test=np.hstack([test_features,final_test1])
print('TF-IDF complete')

classifier_lin = svm.SVC(kernel='linear')
classifier_lin.fit(final_train,train_labels)
print("training completed")
prediction_rbf=classifier_lin.predict(final_test)
print("TF-IDF")
print(classification_report(test_labels, prediction_rbf))
print(accuracy_score(test_labels, prediction_rbf))

print("BNB")
classifier_bnb=BernoulliNB()
classifier_bnb.fit(final_train,train_labels)
print("training completed")
prediction_bnb=classifier_bnb.predict(final_test)
print(classification_report(test_labels, prediction_bnb))
print(accuracy_score(test_labels, prediction_bnb))


