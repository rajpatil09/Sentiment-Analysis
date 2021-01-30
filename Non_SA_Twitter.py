import os
import io
import re
import pickle
import nltk
#from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import random
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify.scikitlearn import SklearnClassifier
import time


regex1 = re.compile('[^a-z|\\s]')
regex2 = re.compile('\\s+')

def eleminateStopWordsString(line):
	#print('original:',line)
	stop_words = set(stopwords.words('english'))
	word_tokens = word_tokenize(line)
	filtered_sentence = ""
	for w in word_tokens:
		if w not in stop_words:
			filtered_sentence += w + " "
	#print('filtered:',filtered_sentence)			
	return filtered_sentence

def eleminateStopWordsList(line):
	#print('original:',line)
	stop_words = set(stopwords.words('english'))
	word_tokens = word_tokenize(line)
	filtered_sentence = []
	for w in word_tokens:
		if w not in stop_words:
			filtered_sentence.append(w)
			#filtered_sentence += w + " "
	#print('filtered:',filtered_sentence)			
	return filtered_sentence

def preprocess(line):
	line = regex1.sub('', line)
	line = regex2.sub(' ', line)
	return line



document=[]
documents=[] 
filepath = "/home/raj/AI/4-SA/Twitter/"
writefile = open(filepath+"Twitter_Data.txt", "a+")
cnt_n=0
cnt_p=0
cnt=0
discard=0
pptr = io.open(filepath+"Data.csv", encoding='latin-1')
reader = pptr.read()

for line in reader.split('\n'):
	words = line.split(",",5)
	label = words[0].lower().strip().replace('"', '')

	if((label == "0") and (cnt_n <50000)):
		
		rline = words[5].lower().strip().replace('"', '')
		rline = preprocess(rline)
		rline = eleminateStopWordsString(rline)
		document = eleminateStopWordsList(rline)						
		documents.append((document, label))
		#wline = label+"^"+rline
		writefile.write(rline+"\n")	#writefile.write(wline+"\n")

		cnt_n+=1
		cnt+=1

	elif((label == "4") and (cnt_p <50000)):

		#print('in:',label)
		rline = words[5].lower().strip().replace('"', '')
		rline = preprocess(rline)
		rline = eleminateStopWordsString(rline)
		document = eleminateStopWordsList(rline)						
		documents.append((document, label))
		#wline = label+"^"+rline
		writefile.write(rline+"\n")	#writefile.write(wline+"\n")		

		cnt_p+=1
		cnt+=1

	else:
		discard+=1

	if((cnt%100000)==0):
			break		

writefile.close()			
print('Considered:',cnt,' Ignored:', discard, ' pos:', cnt_p, ' neg:', cnt_n)



#print(len(documents))
#print(documents[0])

#all_lines=[]  #all lines
all_words=[]   #all words

for line in open(filepath+'Twitter_Data.txt', 'r'):
	word_tokens = word_tokenize(line)
	all_words.append(word_tokens)
	#all_data.append(line)

wordlist=[]
for words in all_words:
	for word in words:
		wordlist.append(word)	#need the duplicates to find the frequency, and order them

print('#feature size (with duplicates):',len(wordlist))

wordlist = nltk.FreqDist(wordlist) #Frequency Distribution
#print('#feature size (unique):',len(wordlist))
print(wordlist.most_common(15))
#print(wordlist["excellent"])
word_features = list(wordlist.keys())[:3000]
print('feature-size limit:',len(word_features))


save_features = open("Twitter/Features.pickle","wb")
pickle.dump(word_features, save_features)
save_features.close()

random.shuffle(documents)	
save_docs = open("Twitter/DocsLabels.pickle","wb")
pickle.dump(documents, save_docs)
save_docs.close()

print('Total Documents: ',len(documents), ' word features:', len(word_features))
'''


getFeatures = open("Twitter/Features.pickle", "rb")
word_features = pickle.load(getFeatures)
getFeatures.close()

getDocs = open("Twitter/DocsLabels.pickle", "rb")
documents = pickle.load(getDocs)
getDocs.close()


def find_features(doc):
    words = set(doc)				        #get unique words in the document
    features = {}							#2D array representing all files, where each row (a file) has all the 3000 words along with value of T (present) or F (absent)
    for w in word_features:
        features[w] = (w in words)
    return features

#print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))
featuresets = [(find_features(rev), category) for (rev, category) in documents]

#for(rev, category) in documents
    #featuresets.append((find_features(rev), category))

training_set = featuresets[:80000]						# 80%

testing_set  = featuresets[80000:]						# 20%

print('Total Documents: ',len(documents), ' Train-Set:', len(training_set), ' Test-Set:',len(testing_set))


start = time.time()
NB_classifier = nltk.NaiveBayesClassifier.train(training_set)
print("NB_classifier accuracy percent:",(nltk.classify.accuracy(NB_classifier, testing_set))*100)
#NB_classifier.show_most_informative_features(15)
end = time.time()
elapsed_time = end - start
print('NB_classifier time taken:', elapsed_time)

start = time.time()
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)
end = time.time()
elapsed_time = end - start
print('MNB_classifier time taken:', elapsed_time)

start = time.time()
BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)
end = time.time()
elapsed_time = end - start
print('BernoulliNB_classifier time taken:', elapsed_time)

start = time.time()
LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)
end = time.time()
elapsed_time = end - start
print('LogisticRegression_classifier time taken:', elapsed_time)

start = time.time()
SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)
end = time.time()
elapsed_time = end - start
print('SGDClassifier_classifier time taken:', elapsed_time)

##SVC_classifier = SklearnClassifier(SVC())
##SVC_classifier.train(training_set)
##print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)

start = time.time()
LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)
end = time.time()
elapsed_time = end - start
print('LinearSVC_classifier time taken:', elapsed_time)

start = time.time()
NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)
end = time.time()
elapsed_time = end - start
print('NuSVC_classifier time taken:', elapsed_time)
'''