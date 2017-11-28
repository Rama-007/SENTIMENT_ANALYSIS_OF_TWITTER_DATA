import os,sys,time,csv,random,re
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.preprocessing import text
import keras
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

from keras.layers import Input,Dense,merge, LSTM
from get_smiley import happy , sad
from linguistic_features import feature_getter,pos_getter,get_wotscore , num_stop_words ,num_punc
from collections import defaultdict
import keras.backend as K
from keras.models import Model
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


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

given_labels=['0','4']

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
	label=line[0]
	words=temp_data.split()
	for i in range(0,len(words)):
		if("http" in words[i]):
			words[i]=''
		else:
			words[i]=words[i].lower()
	#temp_data=' '.join(words)
	data.append((words,label))
	
print('read complete')
random.shuffle(data)
data=data[:10000]

for i in range(0,len(data)):
	temp=data[i][0]
	for j in range(0,len(temp)):
		if temp[j] in acronyms:
			temp[j]=acronyms[temp[j]]
	temp1=' '.join(temp)
	data[i]=(temp1,data[i][1])

for i in range(0,len(data)):
	data[i]=(re.sub('#.*? ','',data[i][0]).split(' '),data[i][1])
print('shuffle complete')
L=len(data)
train_data=[i[0] for i in data[:int(0.8*L)]]
train_labels=[i[1] for i in data[:int(0.8*L)]]
train_features=[]
for k in train_data:
	i=' '.join(k)
	i=re.sub(r'[^\x00-\x7F]+',' ', i)
	train_features.append(np.asarray(get_features(i)))
test_data=[i[0] for i in data[int(0.8*L):]]
test_labels=[i[1] for i in data[int(0.8*L):]]
test_features=[]
for k in test_data:
	i=' '.join(k)
	i=re.sub(r'[^\x00-\x7F]+',' ', i)
	test_features.append(np.asarray(get_features(i)))

X = np.concatenate((train_data, test_data), axis=0)
y = np.concatenate((train_labels, test_labels), axis=0)

vocabulory=np.unique(np.hstack(X))
X_train=[]
X_test=[]
for i in train_data:
	k=' '.join(i)
	X_train.append(text.one_hot(k,len(vocabulory)))
for i in test_data:
	k=' '.join(i)
	X_test.append(text.one_hot(k,len(vocabulory)))

X_train = sequence.pad_sequences(X_train, maxlen=300)
X_test = sequence.pad_sequences(X_test, maxlen=300)

encoder=LabelEncoder()
encoder.fit(train_labels)
encoded_Y=encoder.transform(train_labels)
y_train=np_utils.to_categorical(encoded_Y)

encoder=LabelEncoder()
encoder.fit(test_labels)
encoded_Y=encoder.transform(test_labels)
y_test=np_utils.to_categorical(encoded_Y)

train_features=np.asarray(train_features)
test_features=np.asarray(test_features)

inputt=Input(shape=(300,))
embedding=Embedding(len(vocabulory), 32, input_length=300)(inputt)
conv=Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(embedding)
pol=MaxPooling1D(pool_size=2)(conv)
den=Dense(300, activation='relu')(pol)
#seq_features=Flatten()(den)
seq_features=LSTM(196,dropout_U=0.2, dropout_W=0.2)(embedding)
nb_features=len(train_features[0])
other_features=Input(shape=(nb_features,))

model_final=  merge([seq_features , other_features], mode='concat')
#model_final = Dense(125, activation='relu')(model_final)
model_final = Dense(2, activation='softmax')(model_final)
model_final = Model([inputt, other_features], model_final)

model_final.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])

model_final.fit([X_train,train_features], y_train,validation_data=([X_test,test_features],y_test), epochs=3, batch_size=128, verbose=2)
predict1=model_final.predict([X_test,test_features])
predicted=[]
for i in range(0,len(predict1)):
	predicted.append(np.argmax(predict1[i]))
true_predict=[]
for i in range(0,len(y_test)):
	true_predict.append(np.argmax(y_test[i]))

print(classification_report(true_predict, predicted))
print(accuracy_score(true_predict, predicted))



inputt=Input(shape=(300,))
embedding=Embedding(len(vocabulory), 32, input_length=300)(inputt)
conv=Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(embedding)
pol=MaxPooling1D(pool_size=2)(conv)
den=Dense(300, activation='relu')(pol)
#seq_features=Flatten()(den)
seq_features=LSTM(100,dropout_U=0.2, dropout_W=0.2)(embedding)
#nb_features=len(train_features[0])
#other_features=Input(shape=(0,))

#model_final=  merge([seq_features , other_features], mode='concat')
#model_final = Dense(125, activation='relu')(seq_features)
model_final = Dense(2, activation='softmax')(seq_features)
model_final = Model(inputt, model_final)
model_final.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])

model_final.fit(X_train, y_train,validation_data=(X_test,y_test), epochs=3, batch_size=128, verbose=2)

predict1=model_final.predict(X_test)
predicted=[]
for i in range(0,len(predict1)):
	predicted.append(np.argmax(predict1[i]))
true_predict=[]
for i in range(0,len(y_test)):
	true_predict.append(np.argmax(y_test[i]))

print(classification_report(true_predict, predicted))
print(accuracy_score(true_predict, predicted))

