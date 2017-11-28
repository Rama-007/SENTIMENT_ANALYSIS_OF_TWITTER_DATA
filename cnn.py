import os,sys,time,csv,random
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

X = np.concatenate((train_data, test_data), axis=0)
y = np.concatenate((train_labels, test_labels), axis=0)

vocabulory=np.unique(np.hstack(X))
X_train=[]
X_test=[]
for i in train_data:
	X_train.append(text.one_hot(i,len(vocabulory)))
for i in test_data:
	X_test.append(text.one_hot(i,len(vocabulory)))

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


model = Sequential()
model.add(Embedding(len(vocabulory), 32, input_length=300))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=128, verbose=2)

scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
