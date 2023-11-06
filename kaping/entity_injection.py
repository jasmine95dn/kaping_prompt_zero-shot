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

	# use this model as main model for entity injector
	model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

	# basic prompts
	no_knowledge_prompt = "Please answer this question"
	leading_prompt = "Below are facts in the form of the triple meaningful to answer the questions"

	def sentence_embedding(self, texts: list):
		"""
		Use MPNET to turn all into sentence embeddings
		:param texts:
		:
		:return: embedding in form of numpy.ndarray
		"""
		return MPNetEntityInjector.model.encode(texts)

	def top_k_triple_extractor(self, question, triples, k=10, baseline=False):
		"""
		Retrieve the top k triples of KGs used as context for the question

		:param question: question in form of sentence embeddings 
		:type question: np.ndarray
		:param triples: triples in form of sentence embeddings
		:type triples: np.ndarray
		:param k:
		:type k:
		:param baseline:
		:type baseline:
		:return: 
		"""
		# in case number of triples is fewer than k 
		if len(triples) < k:
			k = len(triples)

		if baseline:
			return random.sample(infos, k)

		# if not the baseline but the top k most similar
		similarities = cosine_similarity(question, triples)
		top_k_indices = np.argsort(similarities[0])[-k:][::-1]

		return [triples[index] for index in top_k_indices]

	def injection(self, question, triples, baseline=False):
		"""

		"""
		if baseline:
			return f"{MPNetEntityInjector.no_knowledge_prompt} {', '.join(triples)} Question: {question} Answer: "
		else:
			return f"{MPNetEntityInjector.leading_prompt} {', '.join(triples)} Question: {question} Answer: "

	def __call__(self, question, triples, k=10, baseline=False):
		"""

		"""
		assert type(question) == list
		assert type(triples) == list

		# use MPNET to turn all into sentence embeddings
		emb_question = self.sentence_embedding(question)
		emb_triples = self.sentence_embedding(triples)

		# retrieve the top k triples
		top_k_triples = self.top_k_triple_extractor(emb_question, emb_triples, k=k, baseline=baseline)

		# create prompt as input
		return self.injection(question, top_k_triples, baseline)

