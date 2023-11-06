"""
This contains a simple script to the pipeline of KAPING
"""

from kaping.entity_extractor import RefinedEntityExtractor
from kaping.entity_verbalization import RebelEntityVerbalizer
from kaping.entity_injection import MPNetInjector


def pipeline(config, question: str, device=-1):
	"""
	Create a pipeline for KAPING
	:param config:
	:param question:
	:param device: choose the device to run this pipeline on, for upgrading to GPU change this to  (default: -1, which means for CPU)
	:return: final prompt as output of KAPING to feed into a QA model
	"""

	# define 3 steps
	extractor = RefinedEntityExtractor(device=device)
	verbalizer = RebelEntityVerbalizer(device=device)
	injector = MPNetInjector(device=device)

	# retrieve entities from given question
	entity_set = extractor(question)

	# entity verbalization
	knowledge_triples = []
	for entity, entity_title in entity_set:
		knowledge_triples.extend(verbalizer(entity, entity_title))

	# entity injection as final prompt as input
	prompt = injector(question, knowledge_triples, k=config['k'], baseline=config['baseline'])

	return prompt

	

