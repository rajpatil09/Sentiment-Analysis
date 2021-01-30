import pickle

'''
def getSentiWords(file):
	for line in open(file, 'r'):       
		line = line.replace('\n', '')
		wdpy = line.split("^")
		lexicon_words.append(wdpy[1])


lexicon_words=[]
lexi=[]
getSentiWords('/home/raj/AI/4-SA/Lexicons/AFINN_labeled.txt')
getSentiWords('/home/raj/AI/4-SA/Lexicons/bing_labeled.txt')
getSentiWords('/home/raj/AI/4-SA/Lexicons/Generalnquirer_labeled.txt')
getSentiWords('/home/raj/AI/4-SA/Lexicons/NRC_labeled.txt')
getSentiWords('/home/raj/AI/4-SA/Lexicons/SentiWordNet_labeled.txt')
getSentiWords('/home/raj/AI/4-SA/Lexicons/Subjectivity_labeled.txt')
getSentiWords('/home/raj/AI/4-SA/Lexicons/vader_labeled.txt')
print('dupli lexi:',len(lexicon_words))
lexi = set(lexicon_words)
print('unique lexi:',len(lexi))
'''

print("IMDB DATASET:")
nsf = open("IMDB/Non_Senti/Features.pickle", "rb")
nsfeatures = pickle.load(nsf)
nsf.close()

sf = open("IMDB/Senti/Senti_Features.pickle", "rb")
sfeatures = pickle.load(sf)
sf.close()

print('Non-Senti:',nsfeatures[0:15])
print('----Senti:',sfeatures[0:15])

present = 0
absent = 0
for word in nsfeatures:
	if word in sfeatures:
		#print('senti:',word)
		present +=1
	else:
		#print('non-senti:',word)
		absent+=1		
print(' common: ',present, ' missing:', absent)		

present = 0
absent = 0
for word in sfeatures:
	if word in nsfeatures:
		#print('senti:',word)
		present +=1
	else:
		#print('missing-senti:',word)
		absent+=1		
print(' common: ',present, ' missing:', absent)		


print("AMAZON DATASET:")
nsf = open("Amazon/Non_Senti/Features.pickle", "rb")
nsfeatures = pickle.load(nsf)
nsf.close()

sf = open("Amazon/Senti/Senti_Features.pickle", "rb")
sfeatures = pickle.load(sf)
sf.close()

print('Non-Senti:',nsfeatures[0:15])
print('----Senti:',sfeatures[0:15])


present = 0
absent = 0
for word in nsfeatures:
	if word in sfeatures:
		#print('senti:',word)
		present +=1
	else:
		#print('non-senti:',word)
		absent+=1		
print(' common: ',present, ' missing:', absent)		

present = 0
absent = 0
for word in sfeatures:
	if word in nsfeatures:
		#print('senti:',word)
		present +=1
	else:
		#print('missing-senti:',word)
		absent+=1		
print(' common: ',present, ' missing:', absent)		


print("TWITTER DATASET:")
nsf = open("Twitter/Non_Senti/Features.pickle", "rb")
nsfeatures = pickle.load(nsf)
nsf.close()

sf = open("Twitter/Senti/Senti_Features.pickle", "rb")
sfeatures = pickle.load(sf)
sf.close()

print('Non-Senti:',nsfeatures[0:15])
print('----Senti:',sfeatures[0:15])


present = 0
absent = 0
for word in nsfeatures:
	if word in sfeatures:
		#print('senti:',word)
		present +=1
	else:
		#print('non-senti:',word)
		absent+=1		
print(' common: ',present, ' missing:', absent)		

present = 0
absent = 0
for word in sfeatures:
	if word in nsfeatures:
		#print('senti:',word)
		present +=1
	else:
		#print('missing-senti:',word)
		absent+=1		
print(' common: ',present, ' missing:', absent)		