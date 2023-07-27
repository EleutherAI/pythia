import re
import string

def _remove_prefixes(aliases):
    # Optimization: Remove any alias that has a strict prefix elsewhere in the list
    # we can do this because if the prefix is acceptable by isgreedy, we can stop looking
    aliases.sort()
    ret = [aliases[0]]
    for alias in aliases[1:]:
        if not alias.startswith(ret[-1]):
            ret.append(alias)
    return ret

def normalize_answer(s):
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


def process_results(doc, results):

    completion = results[0]

    def compute_exact(a_pred, a_gold):
        return int(normalize_answer(a_gold) == normalize_answer(a_pred))

    results = [compute_exact(completion, gold) for gold in _remove_prefixes(doc['answer']['aliases'])]
    return {
        "acc": max(results),
        # "f1": doc['question_id']
        }

def passthrough(arr):
    return arr