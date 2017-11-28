import re,sys
mycompile = lambda pat: re.compile(pat, re.UNICODE)
NormalEyes = r'[:=]'
Wink = r'[;]'
NoseArea = r'(|o|O|-)'
HappyMouths = r'[D\)\]]'
SadMouths = r'[\(\[]'
Tongue = r'[pP]'
OtherMouths = r'[doO/\\]'
Happy_RE =  mycompile( '(\^_\^|' + NormalEyes + NoseArea + HappyMouths + ')')
Sad_RE = mycompile(NormalEyes + NoseArea + SadMouths)

def happy(text):
	h=Happy_RE.search(text)
	if h:
		return 1
	else:
		return 0
def sad(text):
	s=Sad_RE.search(text)
	if s:
		return 1
	else:
		return 0
