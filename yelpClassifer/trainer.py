import sys
import string
import tokenize
import math

punctuation = ["'", ".", "!", "?", ",", "(", ")", "/", "\\", "-", "+", "$", "\t", "*", "&", ":", ""]
num = [str(x) for x in range(10)]

def preProcess(data, vocab):
	data = stripData(data)
	data = data.split("\n")
	data = feature(data, vocab)
	return data


def stripData(data):
	for i in punctuation:
		data =data.replace(i, "")	
	return data.lower()

def buildVocab(data):
	vocab = {}
	for i in data:
		temp = i.split(" ")
		for x in temp:
			if len(x) > 0 and x != '0' and x != '1':
				vocab[x] = 1
	return sorted(vocab)

def feature(data, vocab):
	features = {}
	lineNo = 0;
	for x in data:
		if len(x) == 0:
			break
		lineNo += 1
		sentence = x.split(" ")
		tempDict = {}
		for i in vocab:
			if i in sentence:
				tempDict[i] = 1
			else:
				tempDict[i] = 0
		tempDict["CLASS LABEL"] = oneOrZero(sentence)
		features[lineNo] = tempDict
	return features
		
def oneOrZero(sentence):	
	for i in range(-1*len(sentence)+1, 0):
		#print(sentence)
		if sentence[i*-1] in num:
			return sentence[i*-1]	

def ppFile(data, vocab, fileN):
	for i in range(len(vocab)-1):
		if i == len(vocab)-2:
			fileN.write("CLASS LABEL\n")
		else:
			fileN.write(vocab[i] + ", ")
	for i in data:
		count = 0
		for x in data[i]:
			if count == len(data[i])-1:
				fileN.write(str(data[i][x]) + "\n")
			else:
				fileN.write(str(data[i][x]) + ", ")	
			count +=1	

def classyProbs(data, vocab):
	probabilityGood = {} #two dictionaries: one for the probability that a review is good, one for bad
	probabilityBad = {}
	totalRev = positiveRev(data) #save the total number of good reviews
	total = len(data)	     #and then the total number of reviews as well.
	for x in vocab:
		g = countGood(data, x, 1) #counts the number of reviews where the word x is present, and the sentiment is a 1
		b = countBad(data, x, 1) # counts the number of reviews where the word x is present, and the sentiment is 0
		#print(g, " ", b)
		if g > 0: 	#here, we need to account for the chance of a 'null' entry: that is, a word in the vocab that doesn't exist in the data.
			probabilityGood[x] = (g +1)/(totalRev+2) #this is the normal probability: # of sentences containing the word/ # of sentences total
		else: 		#if there are 0 reviews where x is 1 and sentiment is 1...
			temp = countGood(data, x, 0) #we instead use the opposite probability: that is, p(x) = 1-p(!x)
			probabilityGood[x] = 1- ((temp +1)/(totalRev+2))
		if b > 0:
			probabilityBad[x] = (b +1)/(total-totalRev+2)
		else:
			temp = countBad(data, x, 0)
			probabilityBad[x] = 1- ((temp +1)/(total-totalRev+2))
		#note how we add +1/2 to each probability:
		#This concept is called "Dirichlet Priors". Essentially we add 1/N, where N is the total number of states for a variable (in our case, 2
		#since a vocab word either exists in the sentence, or it does not). This way, when we have a scenario where a word doesn't show up at all
		#in our table, its probability is defaulted to the random chance it is chosen. In this case, 50%, like a coin flip.
	return (probabilityGood, probabilityBad)	

	
def countGood(data, target, num):
	suma = 0
	#print(data)	
	for i in data:
		if data[i][target] == num and data[i]['CLASS LABEL'] == '1':
			suma+=1
	return suma

def countBad(data, target, num):
	suma = 0
	for i in data:
		if data[i][target] == num and data[i]['CLASS LABEL'] == '0':
			suma+=1
	return suma

	

def positiveRev(data):
	count = 0
	for x in data:
		if data[x]['CLASS LABEL'] == "1":
			count += 1
	return count

def priors(prob): #unused
	for i in prob:
		if prob[i] == 0:
			prob[i] = .5
	return prob

def predict(good, bad, data):
	pOne = 0
	pZero = 0
	probY = positiveRev(data)/len(data)
	predicted = {}
	for i in data:
		#print(i)
		for x in data[i]:
			if x != 'CLASS LABEL':
				if data[i][x] == 1: #to predict the sentiment of a sentence, we need to add all of the conditional probabilties.
					pOne += good[x] #we use logs because math. It improves the accuracy because numbers are strange.
					pZero += bad[x]
				#Here we could also add the probabilities for the words that don't show up, ie where data[i][x] == 0
				#however, in our case, it doesn't help the accuracy. A word not showin up doesn't imply anything meaningful.
		if (pOne  + math.log(probY, 10)) > (pZero + math.log(1-probY, 10)):#make the prediction: is the probability of 1 or 0 higher?
			predicted[i] = 1
		else:
			predicted[i] = 0
		pOne = 0
		pZero = 0
	
	return checkAccuracy(predicted, data) #check the accuracy of the prediction using the predicted table.

def checkAccuracy(expected, data):
	actual = {}
	correct = 0
	maximum = 0
	#print(expected, " is this working?")
	for i in data:
		if data[i]["CLASS LABEL"]== '1':	
			actual[i] = 1
		else:
			actual[i] = 0
	for i in expected:
		if expected[i] == actual[i]:
			correct += 1
	#print(expected, " ", actual)
	#print("correctness: ", correct/len(data))
	return correct/len(data)
		
			
	

if __name__ == "__main__":

	#read in the training and testing set via cmd argument	
	training = (open(sys.argv[1], "r")).read()
	testing = (open(sys.argv[2], "r")).read()
	afterTrain = open("preprocessed_train.txt", "w")
	afterTest = open("preprocessed_test.txt", "w")
	results = open("results.txt", "w") #open a file for output: not important


	vocab = buildVocab((stripData(training).split("\n")))#split the training data by line, and build the vocab
	#first, preprocess the data into a data structure for use with the bayes algorithm.
	#Each file is used to create an array of feature vectors representing sentences.
	#Each sentence is represented by a vocab based dictionary. For example, if my vocab is "apple cat dog", and my 
	#sentence is "Domino's sucks", my entry in the structure would look like: {apple: 0, cat: 0, dog: 0, CLASS LABEL: 0}
	#Class label is the sentiment of the review (0 for a bad review, 1 for good) and the 0's for apple, cat, etc show that 
	#the word in the vocab is not present in the sentence. This makes it easy to determine which words can be used to determine
	#the overall probability later. Note #1: None of the words in the actual review are part of the vocabulary: this is essentially
	#the difference between training and testing. In a more realisitic situation, we'll have any number of words in the testing sentence
	#that were also present in the training sentence. Note #2: you only really need to understand the structure of the output here. 
	#The process isn't unique or important.
	trainingData = preProcess(training, vocab)
	testingData = preProcess(testing, vocab)

	ppFile(trainingData, vocab, afterTrain) #saves preprocessed data, not important
	ppFile(testingData, vocab, afterTest)	
	
	#Here, we use bayes algorithm to classify the probabilities for the two sets of data. 
	#The percent accuracy will be higher for the training data, since every word in each sentence is already in the vocabulary.
	#Like I noted above, the same won't be true for the testing set.
	p = classyProbs(trainingData, vocab)
	a = predict(p[0], p[1], trainingData);
	print("Training Set used on Training Set, correctness: ", a)
	b = predict(p[0], p[1], testingData);
	print("Training Set used on Testing Set, correctness: ", b)
	#write results, not important
	results.write("Trained by the training set, tested on the training set, correctness =")
	results.write(str(a))
	results.write("\n")
	results.write("Trained by the training set, tested on the testing set, correctness =")
	results.write(str(b))
	


	
