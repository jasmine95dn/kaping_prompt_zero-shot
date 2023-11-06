"""
This contains the script for a simple off-the-shelf entity extractor, framework used here is ReFInED,
an entity linking framework

"""

from refined.inference.processor import Refined

class RefinedEntityExtractor:

	"""
	For example: "Who is the author of Lady Susan?":
	Result:
	[['Lady Susan', Entity(wikidata_entity_id=Q581180, wikipedia_entity_title=Lady Susan), None]]
	"""

	def __init__(self, device=-1):

		# set extractor as ReFInED extractor
		self.extractor = Refined.from_pretrained(model_name="wikipedia_model_with_numbers", entity_set="wikipedia", device=device)


	def __call__(self, text: str):
		"""

		Expect this entity extractor will return info of the entity in raw text and its respective on Wikipedia and Wikidata
		We only need the entity in raw text, and the Wikipedia info (or indeed after disambiguied by ReFIned) will be the title name of that Entity on Wikipedia page
		
		:param text: Question as text to pass on this extractor
        :return: A list of tuples (entity, entity_wikipedia_title).
		"""

		# in case the question given does not split '?' at the end from the last token,
		# doing this double check to make sure ReFInEd found the entity if that entity stands at the end of the question
		if text.endswith('?') and not text.endswith(' ?'):
			text = text[:-1]+' '+'?'

		# preprocessing: extract entity linking into different spans
		spans = self.extractor.process_text(text)

		# define set of entity
		entity_set = []
		print("***** Entity Extraction *****")
		entity_set = [(span.text, span.predicted_entity.wikipedia_entity_title)
							for span in spans]

		return entity_set
