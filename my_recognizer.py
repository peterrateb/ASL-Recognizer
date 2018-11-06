import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
	""" Recognize test word sequences from word models set
	
	param models: dict of trained models
	{'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
	param test_set: SinglesData object
	return: (list, list)  as probabilities, guesses
	both lists are ordered by the test set word_id
	probabilities is a list of dictionaries where each key a word and value is Log Liklihood
		[{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
			{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
			]
	guesses is a list of the best guess words ordered by the test set word_id
		['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
	"""
	warnings.filterwarnings("ignore", category=DeprecationWarning)
	probabilities = []
	guesses = []
	# TODO implement the recognizer
	#get the number of words (sequences)
	total_words = len(test_set.get_all_sequences())
	#iterate over all words in test_set
	for word_index in range(total_words) :
		prop= {} #initialoze an empty dict 
		best_fit_word = None
		best_prop = float('-inf') #to get the most fit word for a specific sequence
		x , lengths = test_set.get_item_Xlengths(word_index)
		#models : dictionary that word is the key and model is the value
		for word,model in models.items():
			try : 
				logL = model.score(x,lengths)
			except :
				continue
			prop[word] = logL
			if logL>best_prop :
				best_prop = logL
				best_fit_word = word
		
		probabilities.append(prop)
		guesses.append(best_fit_word)
	# return probabilities, guesses
	return probabilities , guesses
	