import jsonlines
import json
from abc import ABC, abstractmethod
import s3fs
import os
from tqdm.auto import tqdm

class JsonlLoader(ABC):
    """Dataset registry class to extract / stream jsonl dataset. 

    Sequences are loaded such that rank `i` gets every ith sequence
    
    """

    def __init__(self, batch_size, world_size = 1, curr_rank = 0):
        self.batch_size = batch_size
        self.world_size = world_size
        self.curr_rank = curr_rank

    @abstractmethod
    def load(self):
        """Function to load / initialize streaming of dataset"""
    
    @abstractmethod
    def __iter__(self):
        """Yields batches of jsonline files, for the given rank"""

    @abstractmethod
    def save(self):
        """Save documents of current rank"""

    def to_jsonl(self, data: dict):
        """Utility function to convert dictionary to a json line. 
        
        Default converter is inefficient

        Returns:
            Json line in binary format
        """
        return json.dumps(data).encode("UTF-8") + b'\n'



class LocalJsonlLoader(JsonlLoader):
    """Loads a jsonl file from local directory"""
    
    def load(self, load_path, save_dir):
        os.makedirs(save_dir, exist_ok = True)
        save_path = os.path.join(save_dir, f"{self.curr_rank}.jsonl")

        self.loader = jsonlines.open(load_path)
        self.writer = open(save_path, mode = "wb") 
    
    def __iter__(self):
        current_batch = []

        iterator = self.loader.iter(type=dict, skip_invalid=True, skip_empty=True)
        for idx, doc in enumerate(iterator):
            if (doc['text'] == ''):
                # print(idx, doc)
                continue
            if not (idx % self.world_size == self.curr_rank):
                # print("Will be skipped by this rank")
                continue
            
            current_batch.append(doc)
            if len(current_batch) >= self.batch_size:
                batch = current_batch[:self.batch_size]
                current_batch = current_batch[self.batch_size:]
                yield batch
    
    def save(self, documents):
        all_data = b''
        for document in documents:
            all_data += self.to_jsonl(document)
        self.writer.write(all_data)
