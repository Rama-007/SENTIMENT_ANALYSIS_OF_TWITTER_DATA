import sys,os
from collections import defaultdict
import re
import nltk
#import requests
from nltk.corpus import stopwords
import string

ssa=defaultdict(int)
for i in open("strongsubj-adj.txt",'rb').readlines():
	i=i.strip().lower()
	ssa[i]=1

wsa=defaultdict(int)
for i in open("weaksubj-adj.txt",'r').readlines():
	i=i.strip().lower()
	wsa[i]=1
ssn=defaultdict(int)
for i in open("strongsubj-noun.txt",'r').readlines():
	i=i.strip().lower()
	ssn[i]=1

wsn=defaultdict(int)
for i in open("weaksubj-noun.txt",'r').readlines():
	i=i.strip().lower()
	wsn[i]=1

sw=defaultdict(int)
for i in open("self_words",'r').readlines():
	i=i.strip().lower()
	sw[i]=1

pp=defaultdict(int)
for i in open("polar-pos.txt",'r').readlines():
	i=i.strip().lower()
	pp[i]=1

pn=defaultdict(int)
for i in open("polar-neg.txt",'r').readlines():
	i=i.strip().lower()
	pn[i]=1

# sp=defaultdict(int)
# for i in open("superlative_list",'r').readlines():
# 	i=i.strip()
# 	sp[i]=1

def feature_getter(text):
	tokenized=nltk.word_tokenize(text)
	s_s_a=0;s_s_n=0;w_s_a=0;w_s_n=0;s_w=0;p_p=0;p_n=0;s_p=0
	for i in tokenized:
		i=i.lower()
		if i in ssa:
			s_s_a+=1
		if i in ssn:
			s_s_n+=1
		if i in wsa:
			w_s_a+=1
		if i in wsn:
			w_s_n+=1
		if i in sw:
			s_w+=1
		if i in pp:
			p_p+=1
		if i in pn:
			p_n+=1
		# if i in sp:
		# 	s_p+=1
	return [s_s_a,s_s_n,w_s_a,w_s_n,s_w,p_p,p_n]

def pos_getter(text):
	tokenized=nltk.word_tokenize(text)
	pos=nltk.pos_tag(tokenized)
	PRP=0;CD=0;MD=0;UH=0;JJS=0;W=0
	for i in pos:
		if(i[1]=="PRP"):
			PRP+=1
		elif(i[1]=="CD"):
			CD+=1
		elif(i[1]=="MD"):
			MD+=1
		elif(i[1]=="UH"):
			UH+=1
		elif(i[1]=="JJS" or i[1]=="RBS"):
			JJS+=1
		elif(i[1]=="WDT" or i[1]=="WP" or i[1]=="WP$" or i[1]=="WRB"):
			W+=1
	return [PRP,CD,MD,UH,JJS,W]

def get_wotscore(url1):
	proxies = {
  'http': 'proxy.iiit.ac.in:8080',
  'https': 'proxy.iiit.ac.in:8080',
}
	url2=re.sub('https://','',url1)
	k=requests.get(url="https://www.mywot.com/en/scorecard/"+url2,proxies=proxies)
	temp=re.search('trust score is .*?/100',k.text).group()
	if(temp==''):
		return 0
	tep=re.sub('trust score is','',temp)
	temp=re.sub('/100','',tep)
	try:
		temp=int(temp)
	except:
		return 0
	return temp

def num_stop_words(text):
	stop=set(stopwords.words('english'))
	tokenized=nltk.word_tokenize(text.decode('utf-8'))
	summ=0
	for word in tokenized:
		if word.lower() in stop:
			summ+=1
	return summ

def num_punc(text):
	pre=lambda x : x in string.punctuation
	return len(list(filter(pre,text)))

