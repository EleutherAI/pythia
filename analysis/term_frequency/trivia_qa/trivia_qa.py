"""
TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension
https://arxiv.org/pdf/1705.03551.pdf

TriviaQA is a reading comprehension dataset containing over 650K question-answer-evidence
triples. TriviaQA includes 95K question-answer pairs authored by trivia enthusiasts
and independently gathered evidence documents, six per question on average, that provide
high quality distant supervision for answering the questions.

Homepage: https://nlp.cs.washington.edu/triviaqa/
"""
import re
import string
import inspect

import datasets
import evaluate

from lm_eval.base import Task, rf
from lm_eval.metrics import mean


_CITATION = """
@InProceedings{JoshiTriviaQA2017,
    author = {Joshi, Mandar and Choi, Eunsol and Weld, Daniel S. and Zettlemoyer, Luke},
    title = {TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension},
    booktitle = {Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics},
    month = {July},
    year = {2017},
    address = {Vancouver, Canada},
    publisher = {Association for Computational Linguistics},
}
"""


class TriviaQA(Task):
    VERSION = 1
    DATASET_PATH = "trivia_qa"
    DATASET_NAME = "unfiltered.nocontext"

    def __init__(self, data_dir=None, cache_dir=None, download_mode=None, DATASET_NAME=None):

        self.EVAL_HARNESS_NAME = "{}_{}".format(
            "long_tail",
            self.DATASET_PATH
        )

        # self.download(data_dir, cache_dir, download_mode)
        self.dataset = datasets.load_dataset(
            path=self.DATASET_PATH,
            name=self.DATASET_NAME,
            cache_dir=cache_dir,
            )
        self._training_docs = None
        self._fewshot_docs = None
        self.metric = evaluate.load("exact_match")

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        return self.dataset["train"]

    def validation_docs(self):
        return datasets.concatenate_datasets([self.dataset["train"], self.dataset["validation"]])

    def test_docs(self):
        raise NotImplementedError()

    def doc_to_text(self, doc):
        return "Q:{question} A:".format(**doc)

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["question"]

    def doc_to_target(self, doc):
        return " " + doc["answer"]["value"]

    def _remove_prefixes(self, aliases):
        # Optimization: Remove any alias that has a strict prefix elsewhere in the list
        # we can do this because if the prefix is acceptable by isgreedy, we can stop looking
        aliases.sort()
        ret = [aliases[0]]
        for alias in aliases[1:]:
            if not alias.startswith(ret[-1]):
                ret.append(alias)
        return ret

    def normalize_answer(self, s):
        """Lower text and remove punctuation, articles and extra whitespace."""
        def remove_articles(text):
            regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
            return re.sub(regex, ' ', text)
        def white_space_fix(text):
            return ' '.join(text.split())
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        def lower(text):
            return text.lower()
        return white_space_fix(remove_articles(remove_punc(lower(s))))


    def construct_requests(self, doc, ctx):

        completion = rf.greedy_until(ctx, "\n")
        return completion

    def process_results(self, doc, results):

        completion = results[0]

        def compute_exact(a_pred, a_gold):
            return int(self.normalize_answer(a_gold) == self.normalize_answer(a_pred))

        results = [compute_exact(completion, gold) for gold in self._remove_prefixes(doc['answer']['aliases'])]

        return {"acc": max(results), "id": doc['question_id']}

    def aggregation(self):

        def _identity(arr):
            return arr

        return {
            # "acc": mean,
            "acc": _identity,
            "id": _identity,
        }

    def higher_is_better(self):
        return {
            "acc": True,
            "id": True
            }