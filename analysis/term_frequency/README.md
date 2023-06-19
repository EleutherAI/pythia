1. Load batchs
2. Convert batches to the detokenized from
3. Run `link_pretraining_data.py` for each checkpoint
4. Run `count_relevant_docs`

qa_entities: entities found in each qa-pair
training_entities: entities found in a training set, in this case each step(batch of 1024 sequences of 2048 token length) fed to the model during training.

output of `count_relevant_docs` is `qa_co_occurrence_split=train.json` which is a list the size of the split dataset (~87.6k rows for train and ~11.3k for validation) that has a count of how many co-occurances of entities in both the qa_entities and the training_entities,

run with `--save_examples` so that we also have `qa_co_occurrence_examples_split=train.pkl` which is a dict of two keys. 
- `examples` points to which lines in the pretraining dataset does the co-occurance originate
- `qa_pairs` that point to what exact entities co-occur

