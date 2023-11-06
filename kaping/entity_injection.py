"""
This contains the script for a simple off-the-shelf entity injector, framework used here is MPNet

Steps:
	1. Pass all extracted triples and the question into MPNet to turn them into sentence embeddings
	2. Use cosine similarity to find the top-k-triples
	3. Inject all of them together to form the prompt

"""
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import random

class MPNetEntityInjector:


	# basic prompts
	no_knowledge_prompt = "Please answer this question"
	leading_prompt = "Below are facts in the form of the triple meaningful to answer the questions"

	def __init__(self, device=-1):

		# use this model as main model for entity injector
		self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

	def sentence_embedding(self, texts: list):
		"""
		Use MPNET to turn all into sentence embeddings
		:param texts: list of texts to turn into sentence embeddings
		:return: embedding in form of numpy.ndarray
		"""
		return self.model.encode(texts)

	def top_k_triple_extractor(self, question: np.ndarray, triples: np.ndarray, k=10, random=False):
		"""
		Retrieve the top k triples of KGs used as context for the question

		:param question: question in form of sentence embeddings
		:param triples: triples in form of sentence embeddings
		:param k: number of triples to retrieve
		:param random: if this is True, retrieve random knowledge 
		:return: list of triples
		"""
		# in case number of triples is fewer than k 
		if len(triples) < k:
			k = len(triples)

		if random:
			return random.sample(infos, k)

		# if not the baseline but the top k most similar
		similarities = cosine_similarity(question, triples)
		top_k_indices = np.argsort(similarities[0])[-k:][::-1]

		return [triples[index] for index in top_k_indices]

	def injection(self, question: str, triples=None, no_knowledge=False):
		"""
		Create prompt based on question and retrieved triples

		:param question: question
		:param triples: list of triples (triples are in string)
		:param no_knowledge: if this is True, combine the knowledge
		:return:
		"""
		if no_knowledge:
				return f"{MPNetEntityInjector.no_knowledge_prompt} Question: {question} Answer: "
		else:
			return f"{MPNetEntityInjector.leading_prompt} {', '.join(triples)} Question: {question} Answer: "

	def __call__(self, question: list, triples: list, k=10, random=False, no_knowledge=False):
		"""
		Retrieve the top k triples of KGs used as context for the question

		:param question: 1 question in form [question]
		:param triples: list of triples
		:param k:
		:param random:
		:param no_knowledge:
		:return:
		"""
		assert type(question) == list
		assert type(triples) == list

		if no_knowledge:
			return self.injection(question, no_knowledge)

		# use MPNET to turn all into sentence embeddings
		emb_question = self.sentence_embedding(question)
		emb_triples = self.sentence_embedding(triples)

		# retrieve the top k triples
		top_k_triples = self.top_k_triple_extractor(emb_question, emb_triples, k=k, random=random)

		# create prompt as input
		return self.injection(question, top_k_triples)

