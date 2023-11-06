
from kaping.model import pipeline
from qa.qa_inference import qa_inference
from qa.qa_evaluate import accuracy, evaluate
from qa.qa_preprocessing import load_dataset
from arguments import k_parser

def main():

	# load arguments
	args = k_parser()

	# load dataset
	dataset = load_dataset(args.input)

	# set up results
	results = []

	# set up evaluated to calculate the accuracy
	evaluated = []

	# ------- run through each question-answer pair and run KAPING
	for qa_pair in dataset:
		# run KAPING to create prompt
		prompt = pipeline(args, qa_pair.question, device=args.device)

		# use inference model to generate predicted answer
		predicted_answer = qa_inference(task=args.inference_task, model_name=args.model_name, 
										prompt=prompt, device=args.device)
		qa_pair.pr_answer = predicted_answer

		# add new qa_pair for output file
		results.append(qa_pair)

		# evaluate to calculate the accuracy
		evaluated.append(evaluated(qa_pair.answer, predicted_answer))

	msg = ""
	if args.no_knowledge:
		msg = " without knowledge"
	else:
		msg = " with random knowledge" if args.random else " using KAPING"

	print(f"Accuracy for infering QA task on {args.model_name}{msg}: {accuracy(evaluated):2f}")

	if args.output:
		print(f"Save results in {args.output}")
		with open(args.output, 'w') as f2w:
			for qa_pair in results:
				f2w.write(f"Question: {qa_pair.question}\nAnswer: {qa_pair.answer}\nPredicted answer: {qa_pair.pr_answer}\n\n")

if __name__ == '__main__':
	main()
