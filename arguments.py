from argparse import ArgumentParser


def k_parser():

	parser = ArgumentParser(description="This contains script for KAPING pipeline")

	parser.add_argument("--input", help="Input json file")
	parser.add_argument("--output", default="./mintaka_predicted.csv" help="Output file, default: ./mintaka_predicted.csv")

	parser.add_argument('--k', type=int, default=10, help="number of triples to retrieve (default: 10)")
	parser.add_argument('--random', action='store_true', help="call this flag to use the baseline of random knowledge instead of KAPING")
	parser.add_argument('--no_knowledge', action='store_true', help="call this flag to just simply add a leading prompt without any knowledge instead of using KAPING")

	parser.add_argument('--inference_task', help="either text2text-generation or text-generation, the others are not suitable for inference")
	parser.add_argument('--model_name', help="""choose in this list: bert-large-uncased, t5-small, t5-base, t5-large for text2text-generation
												gpt2 for text-generation""")

	parser.add_argument('--device', type=int, default=-1, help="whether to use GPU or CPU, default is CPU (-1), use GPU set device from 0")
	

	args = parser.parse_args()
	return args