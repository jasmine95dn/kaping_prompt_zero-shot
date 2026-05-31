import sys
from unittest.mock import patch
from arguments import k_parser


class TestKParser:
    def _parse(self, args):
        with patch.object(sys, "argv", ["run.py"] + args):
            return k_parser()

    def test_defaults(self):
        args = self._parse([])
        assert args.k == 10
        assert args.inference_task == "text2text-generation"
        assert args.model_name == "bert-large-uncased"
        assert args.device == -1
        assert args.random is False
        assert args.no_knowledge is False
        assert args.input is None
        assert args.output is None

    def test_custom_k(self):
        args = self._parse(["--k", "5"])
        assert args.k == 5

    def test_custom_model(self):
        args = self._parse(["--model_name", "t5-base"])
        assert args.model_name == "t5-base"

    def test_custom_inference_task(self):
        args = self._parse(["--inference_task", "text-generation"])
        assert args.inference_task == "text-generation"

    def test_gpu_device(self):
        args = self._parse(["--device", "0"])
        assert args.device == 0

    def test_random_flag(self):
        args = self._parse(["--random"])
        assert args.random is True

    def test_no_knowledge_flag(self):
        args = self._parse(["--no_knowledge"])
        assert args.no_knowledge is True

    def test_input_and_output(self):
        args = self._parse(["--input", "data.json", "--output", "results.csv"])
        assert args.input == "data.json"
        assert args.output == "results.csv"
