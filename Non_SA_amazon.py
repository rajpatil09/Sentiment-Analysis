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



'''
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



documents=[] 
document=[]
cnt=0
threshold = 0
filepath = "/home/raj/AI/4-SA/Amazon/"
#print(filepath)
files = os.listdir(filepath)
for file in files:
	if(os.path.isdir(os.path.join(filepath,file)) == False):
		if(file.find("train") !=-1):
			renamefile = "train"
			threshold = 80000
		else:
			renamefile="test"
			threshold = 20000
		writefile = open(filepath+"Non_Senti/"+renamefile+"_Data.txt", "a+")
		cnt=0
		print(filepath+file)
		for line in open(filepath+file, 'r'):
			#print(line)
			#print('[0]:',words[0])
			#print('[1]:',words[1])
			words = line.split(" ", 1)
			label = words[0]
			rline = words[1].lower().strip()
			rline = preprocess(rline)
			rline = eleminateStopWordsString(rline)
			document = eleminateStopWordsList(rline)						
			documents.append((document, label))
			#wline = label+"^"+rline
			writefile.write(rline+"\n")	#writefile.write(wline+"\n")
			cnt+=1
			if(cnt >= threshold):
				break
		writefile.close()	
		print('lines:', cnt)
		#print(documents)


#random.shuffle(documents)	#no need to shuffle, as the data is provided in train and test format
#print(len(documents))
#print(documents[0])


#all_lines=[]  #all lines
all_words=[]   #all words

for line in open(filepath+"Non_Senti/"+'train_Data.txt', 'r'):
	word_tokens = word_tokenize(line)
	all_words.append(word_tokens)
	#all_data.append(line)

for line in open(filepath+"Non_Senti/"+'test_Data.txt', 'r'):
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


save_features = open("Amazon/Non_Senti/Features.pickle","wb")
pickle.dump(word_features, save_features)
save_features.close()

save_docs = open("Amazon/Non_Senti/DocsLabels.pickle","wb")
pickle.dump(documents, save_docs)
save_docs.close()


'''
getFeatures = open("Amazon/Non_Senti/Features.pickle", "rb")
word_features = pickle.load(getFeatures)
getFeatures.close()

getDocs = open("Amazon/Non_Senti/DocsLabels.pickle", "rb")
documents = pickle.load(getDocs)
getDocs.close()

print('Total Documents: ',len(documents), ' word features:', len(word_features))



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

testing_set  = featuresets[:20000]						# set that we'll train our classifier with

training_set = featuresets[20000:]						# set that we'll test against.

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

#start = time.time()
#NuSVC_classifier = SklearnClassifier(NuSVC())
#NuSVC_classifier.train(training_set)
#print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)
#end = time.time()
#elapsed_time = end - start
#print('NuSVC_classifier time taken:', elapsed_time)


