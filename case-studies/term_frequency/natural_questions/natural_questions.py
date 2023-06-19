"""
Natural Questions: a Benchmark for Question Answering Research
https://storage.googleapis.com/pub-tools-public-publication-data/pdf/1f7b46b5378d757553d3e92ead36bda2e4254244.pdf

The Natural Questions (NQ) corpus is a question-answering dataset that contains
questions from real users and requires QA systems to read and comprehend an entire
Wikipedia article that may or may not contain the answer to the question. The
inclusion of real user questions, and the requirement that solutions should read
an entire page to find the answer, cause NQ to be a more realistic and challenging
task than prior QA datasets.

TODO: NaturalQS has a *really* large train set that huggingface just automatically
downloads even if you dont use it. we should try and only download the val set and
not even bother with the train set.

Homepage: https://ai.google.com/research/NaturalQuestions
"""
import datasets

from lm_eval.base import Task, rf
from itertools import islice
from lm_eval.metrics import mean


_CITATION = """
@article{47761,
    title={Natural Questions: a Benchmark for Question Answering Research},
    author={Tom Kwiatkowski and Jennimaria Palomaki and Olivia Redfield and Michael Collins and Ankur Parikh and Chris Alberti and Danielle Epstein and Illia Polosukhin and Matthew Kelcey and Jacob Devlin and Kenton Lee and Kristina N. Toutanova and Llion Jones and Ming-Wei Chang and Andrew Dai and Jakob Uszkoreit and Quoc Le and Slav Petrov},
    year={2019},
    journal={Transactions of the Association of Computational Linguistics}
}
"""


class NaturalQs(Task):
    VERSION = 0
    DATASET_PATH = "natural_questions"
    DATASET_NAME = None

    def __init__(self, data_dir=None, cache_dir=None, download_mode=None, DATASET_NAME=None):

        # self.download(data_dir, cache_dir, download_mode)
        self.dataset = datasets.load_dataset(
            path=self.DATASET_PATH,
            name=self.DATASET_NAME,
            cache_dir=cache_dir,
            )
        self._training_docs = None
        self._fewshot_docs = None


    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        # Cache training for faster few-shot.
        # Data is too large to fit in memory.
        if self._training_docs is None:
            self._training_docs = list(self.dataset["train"])
        return self._training_docs

    def validation_docs(self):
        return self.dataset["validation"]

    def fewshot_examples(self, k, rnd):
        # Data is too large to fit in memory. We just sample from the first bit.
        if self._training_docs is None:
            self._training_docs = list(islice(self.training_docs(), 0, 100000))

        return rnd.sample(self._training_docs, k)

    def doc_to_text(self, doc):
        return "Q: " + doc["question"]["text"] + "\n\n" + "A:"

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["question"]["text"]

    def doc_to_target(self, doc):
        # There's a short answer and a long answer. Based on the paper, I'm using the long answer.
        # short_answer = doc["annotations"]["short_answers"][0]["text"]
        long_answer_start = doc["annotations"]["long_answer"][0]["start_token"]
        long_answer_end = doc["annotations"]["long_answer"][0]["end_token"]
        long_answer_span = doc["document"]["tokens"]["token"][
            long_answer_start:long_answer_end
        ]
        long_answer_is_html = doc["document"]["tokens"]["is_html"][
            long_answer_start:long_answer_end
        ]
        long_answer_chars = [
            tok
            for (tok, is_html) in zip(long_answer_span, long_answer_is_html)
            if not is_html
        ]
        long_answer = " ".join(long_answer_chars)
        return long_answer  # Replace with short_answer[0] for short answer

    def construct_requests(self, doc, ctx):

        completion = rf.greedy_until(ctx, "\n")
        return completion

    def process_results(self, doc, results):
        completion = results[0]
        gold = self.doc_to_target(doc)
        acc = 1.0 if completion == gold else 0.0

        return {"acc": acc}

    def aggregation(self):
        return {
            "acc": mean,
        }

    def higher_is_better(self):
        return {"acc": True}