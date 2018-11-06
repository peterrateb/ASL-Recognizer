import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
	'''
	base class for model selection (strategy design pattern)
	'''
	
	def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
				n_constant=3,
				min_n_components=2, max_n_components=10,
				random_state=14, verbose=False):
		self.words = all_word_sequences
		self.hwords = all_word_Xlengths
		self.sequences = all_word_sequences[this_word]
		self.X, self.lengths = all_word_Xlengths[this_word]
		self.this_word = this_word
		self.n_constant = n_constant
		self.min_n_components = min_n_components
		self.max_n_components = max_n_components
		self.random_state = random_state
		self.verbose = verbose
	
	def select(self):
		raise NotImplementedError
	
	def base_model(self, num_states):
		# with warnings.catch_warnings():
		warnings.filterwarnings("ignore", category=DeprecationWarning)
		# warnings.filterwarnings("ignore", category=RuntimeWarning)
		try:
			hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
									random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
			if self.verbose:
				print("model created for {} with {} states".format(self.this_word, num_states))
			return hmm_model
		except:
			if self.verbose:
				print("failure on {} with {} states".format(self.this_word, num_states))
			return None
	
	
class SelectorConstant(ModelSelector):
	""" select the model with value self.n_constant
	
	"""
	
	def select(self):
		""" select based on n_constant value
	
		:return: GaussianHMM object
		"""
		best_num_components = self.n_constant
		return self.base_model(best_num_components)
	
	
class SelectorBIC(ModelSelector):
	""" select the model with the lowest Bayesian Information Criterion(BIC) score
	
	http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
	Bayesian information criteria: BIC = -2 * logL + p * logN
	"""
	#logl : log liklihood
	#p : number of parameters
	#N : number of data points
	#the best model is the lowest BIC score
	def select(self):
		""" select the best model for self.this_word based on
		BIC score for n between self.min_n_components and self.max_n_components
	
		:return: GaussianHMM object
		"""
		#warnings.filterwarnings("ignore", category=DeprecationWarning)
	
		# TODO implement model selection based on BIC scores
		best_score = float('inf') # +ve inf because need to get the minimum
		best_num_components = None
		
		for n_component in range(self.min_n_components , self.max_n_components+1):
			# BIC score = -2 * logL + p * logN
			#to calculate parameters refer to this discussion : https://stats.stackexchange.com/questions/12341/number-of-parameters-in-markov-model
			# p = ( num_states ** 2 ) + ( 2 * num_states * num_data_points ) -1
			#The hmmlearn library may not be able to train or score all models. 
			#so I Implemented try/except contructs as necessary to eliminate non-viable models from consideration.
			try :
				model = GaussianHMM(n_component , n_iter=1000).fit(self.X, self.lengths)
				logL = model.score(self.X, self.lengths)
				num_data_points = len(self.X[0])
				Parameters = n_component**2 + ( 2 * n_component * num_data_points ) - 1
				score = -2 * logL + Parameters * np.log(num_data_points)
				if score < best_score :
					best_score = score
					best_num_components = n_component
			except :
				pass
		if 	best_num_components != None :
			return self.base_model(best_num_components)
		else :
			return self.base_model(self.n_constant)
	
	
class SelectorDIC(ModelSelector):
	''' select best model based on Discriminative Information Criterion
	
	Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
	Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
	http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
	https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
	DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
	'''
	
	def select(self):
		warnings.filterwarnings("ignore", category=DeprecationWarning)
	
		# TODO implement model selection based on DIC scores
		# the best DIC sccore is the highest one
		# high liklihood to the currrent word and low total liklihood to the oher words
		# log(p(x(i)) = logL == model.score(...)
		# M = number of medels or number of total words that we compares with .
		
		best_score = float('-inf')
		best_num_components = None
		
		# get logL for all models = SUM(log(P(X(all))
		
		M = len(self.hwords.keys()) #number of models
		for n_component in range(self.min_n_components , self.max_n_components+1):
			#The hmmlearn library may not be able to train or score all models. 
			#so I Implemented try/except contructs as necessary to eliminate non-viable models from consideration.
			try :
				model = GaussianHMM(n_component , n_iter=1000).fit(self.X, self.lengths)
				logL = model.score(self.X, self.lengths)
				# get logL for all models = SUM(log(P(X(all))
				sum_logL = 0
				for word in self.hwords.keys() :
					try :
						word_X, word_lengths = self.hwords[word]
						word_model = GaussianHMM(n_component , n_iter=1000).fit(word_X, word_lengths)
						word_logL = model.score(word_X, word_lengths)
						sum_logL+=word_logL
					except :
						sum_logL+=0
				score = logL - (1/(M-1)) * (sum_logL - logL)
				if score > best_score :
					best_score = score
					best_num_components = n_component
			except :
				pass
		
			
		if 	best_num_components != None :
			return self.base_model(best_num_components)
		else :
			return self.base_model(self.n_constant)
	
	
	
class SelectorCV(ModelSelector):
	''' select best model based on average log Likelihood of cross-validation folds
	
	'''
	
	def select(self):
		warnings.filterwarnings("ignore", category=DeprecationWarning)
	
		# TODO implement model selection using CV
		
		best_score = float('-inf')
		best_num_components = None
		nsplits = 3 #default n splits for Kfold
		
		for n_component in range(self.min_n_components , self.max_n_components+1): 
			total_logL = [] # to get average log L of the model
			# Cannot have number of splits n_splits=3 greater than the number of samples: 2
			if(len(self.sequences) < nsplits):
				break
			split_method = KFold(random_state=self.random_state, n_splits=nsplits)
				
			for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
				'''
					In order to run hmmlearn training using the X,lengths tuples on the new folds, subsets must be combined
					based on the indices given for the folds.
					A helper utility has been provided in the asl_utils module named combine_sequences for this purpose.
				'''
				train_X , train_legnths = combine_sequences(cv_train_idx , self.sequences)
				test_X , test_legnths = combine_sequences(cv_test_idx , self.sequences)
				try :
					model = GaussianHMM(n_component , n_iter=1000).fit(self.X, self.lengths)
					logL = model.score(test_X, test_legnths)
					total_logL.append(logL)
	
				except :
					continue
			
			#get the avg of logL
			avg_logL_score = np.mean(total_logL)
			if avg_logL_score > best_score :
				best_score = avg_logL_score
				best_num_components = n_component
					
		if 	best_num_components != None :
			return self.base_model(best_num_components)
		else :
			return self.base_model(self.n_constant)
	
	
	