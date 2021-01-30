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

def getSentiWords(file):
	for line in open(file, 'r'):       
		line = line.replace('\n', '')
		wdpy = line.split("^")
		lexicon_words.append(wdpy[1])


def eleminateNonSentiString(line):
	word_tokens = word_tokenize(line)
	filtered_sentence = ""
	for w in word_tokens:
		if w in lexi:
			filtered_sentence += w + " "
	return filtered_sentence

def eleminateNonSentiList(line):
	word_tokens = word_tokenize(line)
	filtered_sentence = []
	for w in word_tokens:
		if w in lexi:
			filtered_sentence.append(w)
	return filtered_sentence


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

lexicon_words=[]
lexi=[]
getSentiWords('/home/raj/AI/4-SA/Lexicons/AFINN_labeled.txt')
getSentiWords('/home/raj/AI/4-SA/Lexicons/bing_labeled.txt')
getSentiWords('/home/raj/AI/4-SA/Lexicons/Generalnquirer_labeled.txt')
getSentiWords('/home/raj/AI/4-SA/Lexicons/NRC_labeled.txt')
getSentiWords('/home/raj/AI/4-SA/Lexicons/Subjectivity_labeled.txt')
getSentiWords('/home/raj/AI/4-SA/Lexicons/vader_labeled.txt')
#getSentiWords('/home/raj/AI/4-SA/Lexicons/SentiWordNet_labeled.txt')
print('dupli lexi:',len(lexicon_words))
lexi = set(lexicon_words)
print('unique lexi:',len(lexi))

regex1 = re.compile('[^a-z|\\s]')
regex2 = re.compile('\\s+')

document=[]
documents=[]#dictionary having key-value pair
path = "/home/raj/AI/4-SA/IMDB/"
folders = os.listdir(path)
cnt=0
for folder in folders:  
	if((folder == "train") or (folder == "test")):
		if os.path.isdir(os.path.join(path,folder)):
			writefile = open("/home/raj/AI/4-SA/IMDB/Senti/Senti_"+folder+"_Data.txt", "a+")
			subfolders = os.listdir(os.path.join(path,folder))
			for subfolder in subfolders:  
				if os.path.isdir(os.path.join(path,folder,subfolder)):
					filepath = os.path.join(path,folder,subfolder)+"/"
					if(subfolder != "unsup"):
						cnt=0
						#print(filepath)
						files = os.listdir(filepath)
						for file in files:
							#print(file)
							readfile = io.open(filepath+file, encoding='latin-1')
							rline=readfile.read().replace('\n', '')                    #read the entire file in one Go
							rline = rline.lower().strip()
							
							rline = preprocess(rline)
							rline = eleminateStopWordsString(rline)
							document = eleminateStopWordsList(rline)						
							rline = eleminateNonSentiString(rline)
							document = eleminateNonSentiList(rline)						

							documents.append((document, subfolder))
							#wline = subfolder+"^"+rline
							writefile.write(rline+"\n")	#writefile.write(wline+"\n")
							readfile.close()	           
							#break				
							cnt+=1
					print('filepath:',filepath,' count:',cnt)	
			writefile.close()				


#random.shuffle(documents)	#no need to shuffle, as the data is provided in train and test format
#print(len(documents))
#print(documents[0])

#all_lines=[]  #all lines
all_words=[]   #list of word-lists

#reviews_train = []     #list/1D-Array
for line in open('/home/raj/AI/4-SA/IMDB/Senti/Senti_train_Data.txt', 'r'):
	#reviews_train.append(line)
	word_tokens = word_tokenize(line)
	all_words.append(word_tokens)
	#all_data.append(line)

#reviews_test = []	   #list/1D-Array
for line in open('/home/raj/AI/4-SA/IMDB/Senti/Senti_test_Data.txt', 'r'):
	#reviews_test.append(line)
	word_tokens = word_tokenize(line)
	all_words.append(word_tokens)
	#all_data.append(line)

#print('train len:',len(reviews_train),' and test len:',len(reviews_test), ' word-array list:', len(all_words))

wordlist=[]  #list of words
for words in all_words:
	for word in words:
		wordlist.append(word)	

print('#feature size (with duplicates):',len(wordlist))


wordlist = nltk.FreqDist(wordlist) #Frequency Distribution
print('#feature size (unique):',len(wordlist))
print(wordlist.most_common(15))
#print(wordlist["excellent"])
word_features = list(wordlist.keys())[:3000]
print('feature-size limit:',len(word_features))


save_features = open("IMDB/Senti/Senti_Features.pickle","wb")
pickle.dump(word_features, save_features)
save_features.close()

save_docs = open("IMDB/Senti/Senti_DocsLabels.pickle","wb")
pickle.dump(documents, save_docs)
save_docs.close()

'''
getFeatures = open("IMDB/Senti/Senti_Features.pickle", "rb")
word_features = pickle.load(getFeatures)
getFeatures.close()

getDocs = open("IMDB/Senti/Senti_DocsLabels.pickle", "rb")
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

training_set = featuresets[:25000]						# set that we'll train our classifier with

testing_set = featuresets[25000:]						# set that we'll test against.


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
