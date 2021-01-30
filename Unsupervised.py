
import pickle

lexicon_wp={}

dups = 0

def getSentiWords(file):
	
	global dups
	ignore=0

	for line in open(file, 'r'):       
		line = line.replace('\n', '')
		wdpy = line.split("^")
		key = wdpy[1]
		polarity = wdpy[0].strip()		

		#print(key,'<->', polarity)

		if key in lexicon_wp.keys():
			dups = dups + 1
		else:
			lexicon_wp[key] = polarity		

	return ignore 			

'''		if((polarity == "neutral") or (polarity == "both")): 
			ignore +=1
		else:
			#lexicon_wp[key] = polarity
		 
			if key in lexicon_wp.keys():
				dups = dups + 1
			else:
				lexicon_wp[key] = polarity				 
'''

def check_polarity(doc):
	pos = 0
	neg = 0
	words = set(doc)				         
	for word in words:
		if word in lexicon_wp.keys():
			val = lexicon_wp.get(word)
			#print(word,'<->', val)
			
			if((val == "positive") or (val == "__label__2") or (val == "4")):           #imdb, amazon, twitter positive labels
				pos+=1
			elif((val == "negative") or (val == "neutral") or (val == "both")  or (val == "__label__1") or (val == "0")):
				neg +=1
			else:
				print(word,'<->', val)
			'''if((val == "positive") or (val == "__label__2") or (val == "4")):           #imdb, amazon, twitter positive labels
				pos+=1
			elif((val == "negative") or (val == "__label__1") or (val == "0")):
				neg +=1				'''
	#print('pos', pos,' neg',neg)	

	if(pos >= neg):
		return "pos"
	else:
		return "neg"


def calc_accuracy(documents):

	match =0
	diff = 0
	#cnt = 0

	for(rev, category) in documents:

		if((category == "__label__2") or (category == "4")):                       #amazon, twitter positive labels
			category = "pos"
		elif((category == "__label__1") or (category == "0")):	
			category = "neg"

		label = check_polarity(rev)
		#print('label:', label, ' category:',category, ' rev:', rev)

		if(label == category):
			match +=1
		else:
			diff +=1

		#cnt +=1
		
		#if(cnt == 10):	
		#	break		
		#print('matched:', match, ' different:',diff)

	accuracy = (match*100)/(match+diff)
	print('accuracy:',accuracy)


def add_Lexicons(documents):


	ign = getSentiWords('/home/raj/AI/4-SA/Lexicons/AFINN_labeled.txt')       
	print('For AFINN len=',len(lexicon_wp), ' duplicates:', dups,' ignored:', ign)
	calc_accuracy(documents)

	ign = getSentiWords('/home/raj/AI/4-SA/Lexicons/Generalnquirer_labeled.txt') 
	print('For Generalnquirer len=',len(lexicon_wp), ' duplicates:', dups,' ignored:', ign)
	calc_accuracy(documents)

	ign = getSentiWords('/home/raj/AI/4-SA/Lexicons/vader_labeled.txt')  
	print('For VADER len=',len(lexicon_wp), ' duplicates:', dups,' ignored:', ign)
	calc_accuracy(documents)
	
	ign = getSentiWords('/home/raj/AI/4-SA/Lexicons/bing_labeled.txt')   
	print('For Bing len=',len(lexicon_wp), ' duplicates:', dups,' ignored:', ign)
	calc_accuracy(documents)
	
	ign = getSentiWords('/home/raj/AI/4-SA/Lexicons/Subjectivity_labeled.txt')
	print('For Subjectivity len=',len(lexicon_wp), ' duplicates:', dups,' ignored:', ign)
	calc_accuracy(documents)

	ign = getSentiWords('/home/raj/AI/4-SA/Lexicons/NRC_labeled.txt')  
	print('For NRC len=',len(lexicon_wp), ' duplicates:', dups,' ignored:', ign)
	calc_accuracy(documents)

	#getSentiWords('/home/raj/AI/4-SA/Lexicons/SentiWordNet_labeled.txt')
	#print('For SentiWordNet len=',len(lexicon_wp), ' duplicates:', dups)
	#calc_accuracy(documents)


'''
#IMDB DATASET
documents=[]
lexicon_wp={}
dups=0
getDocs = open("IMDB/Non_Senti/DocsLabels.pickle", "rb")
documents = pickle.load(getDocs)
getDocs.close()
print("-----------------------------IMDB DATASET-----------------------------", len(documents))
add_Lexicons(documents)
'''

'''
#AMAZON DATASET
documents=[]
lexicon_wp={}
dups=0
getDocs = open("Amazon/Non_Senti/DocsLabels.pickle", "rb")
documents = pickle.load(getDocs)
getDocs.close()
print("-----------------------------Amazon DATASET-----------------------------", len(documents))
add_Lexicons(documents)
'''

'''
#TWITTER DATASET
documents=[]
lexicon_wp={}
dups=0
getDocs = open("Twitter/Non_Senti/DocsLabels.pickle", "rb")
documents = pickle.load(getDocs)
getDocs.close()
print("-----------------------------Twitter DATASET-----------------------------", len(documents))
add_Lexicons(documents)
'''




'''AMAZON
getSentiWords('/home/raj/AI/4-SA/Lexicons/AFINN_labeled.txt')       
print('For AFINN len=',len(lexicon_wp), ' duplicates:', dups)
calc_accuracy(documents)

getSentiWords('/home/raj/AI/4-SA/Lexicons/Generalnquirer_labeled.txt') 
print('For Generalnquirer len=',len(lexicon_wp), ' duplicates:', dups)
calc_accuracy(documents)

getSentiWords('/home/raj/AI/4-SA/Lexicons/vader_labeled.txt')  
print('For VADER len=',len(lexicon_wp), ' duplicates:', dups)
calc_accuracy(documents)

getSentiWords('/home/raj/AI/4-SA/Lexicons/Subjectivity_labeled.txt')
print('For Subjectivity len=',len(lexicon_wp), ' duplicates:', dups)
calc_accuracy(documents)

getSentiWords('/home/raj/AI/4-SA/Lexicons/bing_labeled.txt')   
print('For Bing len=',len(lexicon_wp), ' duplicates:', dups)
calc_accuracy(documents)

getSentiWords('/home/raj/AI/4-SA/Lexicons/NRC_labeled.txt')  
print('For NRC len=',len(lexicon_wp), ' duplicates:', dups)
calc_accuracy(documents)

#getSentiWords('/home/raj/AI/4-SA/Lexicons/SentiWordNet_labeled.txt')
#print('For SentiWordNet len=',len(lexicon_wp), ' duplicates:', dups)
#calc_accuracy(documents)
'''


'''IMDB

getSentiWords('/home/raj/AI/4-SA/Lexicons/AFINN_labeled.txt')       
print('For AFINN len=',len(lexicon_wp), ' duplicates:', dups)
calc_accuracy(documents)

getSentiWords('/home/raj/AI/4-SA/Lexicons/Generalnquirer_labeled.txt') 
print('For Generalnquirer len=',len(lexicon_wp), ' duplicates:', dups)
calc_accuracy(documents)

getSentiWords('/home/raj/AI/4-SA/Lexicons/vader_labeled.txt')  
print('For VADER len=',len(lexicon_wp), ' duplicates:', dups)
calc_accuracy(documents)

getSentiWords('/home/raj/AI/4-SA/Lexicons/Subjectivity_labeled.txt')
print('For Subjectivity len=',len(lexicon_wp), ' duplicates:', dups)
calc_accuracy(documents)

getSentiWords('/home/raj/AI/4-SA/Lexicons/bing_labeled.txt')   
print('For Bing len=',len(lexicon_wp), ' duplicates:', dups)
calc_accuracy(documents)

getSentiWords('/home/raj/AI/4-SA/Lexicons/NRC_labeled.txt')  
print('For NRC len=',len(lexicon_wp), ' duplicates:', dups)
calc_accuracy(documents)


#getSentiWords('/home/raj/AI/4-SA/Lexicons/SentiWordNet_labeled.txt')
#print('For SentiWordNet len=',len(lexicon_wp), ' duplicates:', dups)
#calc_accuracy(documents)


#71.218
#getSentiWords('/home/raj/AI/4-SA/Lexicons/AFINN_labeled.txt')           #66.06
#getSentiWords('/home/raj/AI/4-SA/Lexicons/Generalnquirer_labeled.txt')  #67.702
#getSentiWords('/home/raj/AI/4-SA/Lexicons/vader_labeled.txt')           #66.742
#getSentiWords('/home/raj/AI/4-SA/Lexicons/bing_labeled.txt')            #68.174
#getSentiWords('/home/raj/AI/4-SA/Lexicons/Subjectivity_labeled.txt')    #71.218
#getSentiWords('/home/raj/AI/4-SA/Lexicons/NRC_labeled.txt')             #71.136
 
#getSentiWords('/home/raj/AI/4-SA/Lexicons/AFINN_labeled.txt')           #66.06
#getSentiWords('/home/raj/AI/4-SA/Lexicons/bing_labeled.txt')            #69.398
#getSentiWords('/home/raj/AI/4-SA/Lexicons/Subjectivity_labeled.txt')    #69.982 
#getSentiWords('/home/raj/AI/4-SA/Lexicons/NRC_labeled.txt')             #70.218 
#getSentiWords('/home/raj/AI/4-SA/Lexicons/vader_labeled.txt')           #70.398
#getSentiWords('/home/raj/AI/4-SA/Lexicons/Generalnquirer_labeled.txt')  #70.442 
##getSentiWords('/home/raj/AI/4-SA/Lexicons/SentiWordNet_labeled.txt')   #68.674
'''





