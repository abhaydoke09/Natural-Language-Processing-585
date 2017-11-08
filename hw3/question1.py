import re
def readTweets():
	f = open('tweets.txt','rb')
	lines = f.readlines()
	f.close()
	result = []
	for line in lines:
		line = line.lower()
		line = re.sub(r'[^a-zA-Z0-9@# \']','',line)
		words = line.split()
		while '' in words:
			words.remove('')
		result.append(words)
	return result





