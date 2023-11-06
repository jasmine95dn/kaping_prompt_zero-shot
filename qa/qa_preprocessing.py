"""
This contains a simple script to load the Mintaka dataset in json-file
"""


import json


class Pair:

    def __init__(self, question: str, entities: list, answer: str, q_type: str):
        """
        This includes a pair of question-answer in Mintaka dataset
        
        """
        self.question = question
        self.entities = entities
        self.answer = answer
        self.q_type = q_type
        self.pr_answer = None


def load_dataset(json_file: str):
    """
    Load the Mintaka data set
    :param json_file: path to Mintaka data
    :return: list of Pair (question, answer)
    """

    qa_pairs = []

    with open(json_file) as f:
        dataset = json.load(f)

    for data in dataset:
        qa_pairs.append(Pair(question=data['question'], 
                              entities=[d['mention'] for d in data['questionEntity']],
                              answer=data['answer'],
                              q_type=data['complexityType']  )
                        )

    return qa_pairs
    



