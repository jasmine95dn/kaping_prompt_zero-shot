
from transformers import pipeline


def qa_inference(task, model_name, prompt, device=-1):
	"""
	Only use pretrained model (without any extra finetuning on any dataset)
	The idea is to pass a whole sequence into the pipeline in form of 
	 "<prompt> question: <question> answer: ""

	Tasks used are text generations/text2text generations depending on how experimented models are supported
	on Hugging Face

	This requires model to learn from input to continue generate text that is suitable for given prompt as input

	"""

	# for bert-large-uncased, t5-small, t5-base, t5-large
	if task == "text2text-generation":
		print("This task is used for bert-large-uncased and t5 models")
		qa_pipeline = pipeline(task, model=model_name, device=device)
		answer = qa_pipeline(prompt)
		return answer[0]['generated_text']

	# for  gpt2
	# In this inference, as huggingface pipeline does not support text2text-generation for gpt2,
	# hence text-generation was used instead
	elif task == "text-generation":
		print("This task is used for GPT2 model")
		qa_pipeline = pipeline(task, model=model_name, max_new_tokens=200, device=device)
		answer = qa_pipeline(prompt)
		return answer[0]['generated_text'].split('Answer: ', 1)
			
		





