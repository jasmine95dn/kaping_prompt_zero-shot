"""
This contains the script for a simple off-the-shelf entity verbalizer, framework used here is Rebel

Steps:
	1. For each entity, find its corresponding infos from the Wikipedia pages (using entity title for searching)
	2. Use Rebel (state-of-the-art models for Relation Extractions in different datasets) to extract relations from extracted texts

"""
import requests
from bs4 import BeautifulSoup
from transformers import pipeline

class RebelEntityVerbalizer:

	"""

	"""

	def __init__(self, task='text2text-generation', model_name='Babelscape/rebel-large', device=-1):

		self.extractor = pipeline(task, model=model_name, tokenizer=model_name, device=device)


	def _extract_triplets(self, text):
		"""
		Use the same code presented on Rebel github page

		"""

		triplets = []
	    relation, subject, relation, object_ = '', '', '', ''
	    text = text.strip()
	    current = 'x'
	    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
	        if token == "<triplet>":
	            current = 't'
	            if relation != '':
	                triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
	                relation = ''
	            subject = ''
	        elif token == "<subj>":
	            current = 's'
	            if relation != '':
	                triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
	            object_ = ''
	        elif token == "<obj>":
	            current = 'o'
	            relation = ''
	        else:
	            if current == 't':
	                subject += ' ' + token
	            elif current == 's':
	                object_ += ' ' + token
	            elif current == 'o':
	                relation += ' ' + token
	    if subject != '' and relation != '' and object_ != '':
	        triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
	    return triplets


	def text_relation(self, text):
		"""
		Extract the triples of relations based on given text,
		Text could be a sentence, a short paragraph (there are limits in the number of tokens)

		"""
		# We need to use the tokenizer manually since we need special tokens.
		extracted_text = self.extractor.tokenizer.batch_decode([triplet_extractor(text, return_tensors=True, return_text=False)[0]["generated_token_ids"]])

		extracted_triplets = self._extract_triplets(extracted_text[0])
		return extracted_triplets

	def _get_wikipedia_paragraph(self, entity, entity_title=None):

		"""
		Extract data from the Wikipedia page of the entity based on its entity title
		Use the function text_relation() to build the knowledge graphs for triples "(subj, relation, obj)"


		"""

		infos = []
		
		if not entity_title:
			entity_title = entity
		
		url = f"https://en.wikipedia.org/wiki/{entity_title}"
		
		response = requests.get(url)
		if response.status_code == 200:
			soup = BeautifulSoup(response.text, "html.parser")
			paragraphs = soup.find_all("p")
			if paragraphs:
				for paragraph in paragraphs:
					paragraph = paragraph.get_text()
					relations = text_relation(paragraph)
					infos.extend(relations)
			else:
				print("No paragraphs")
		else:
			print("No data")
		return infos

	def __call__(self, entity, entity_title=None):
		"""
		Output:
		infos = ['(Black Eyed Peas, has part, will.i.am)',
		 '(Black Eyed Peas, has part, apl.ap)',
		 '(Black Eyed Peas, has part, Taboo)',
		 '(Black Eyed Peas, has part, will.i.am)',
		 '(Black Eyed Peas, has part, apl.ap)',
		 '(Black Eyed Peas, has part, Taboo)',
		 '(Black Eyed Peas, has part, Stacy Ferguson)',
		 '(Black Eyed Peas, record label, Epic Records)',
		 ....,]
		"""

		print("***** Verbalization *****")
		return self._get_wikipedia_paragraph(entity, entity_title)



