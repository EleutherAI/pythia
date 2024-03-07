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

    @abstractmethod
    def close(self):
        """Perform cleanups and close any open readers / writers"""
    
    @abstractmethod
    def __len__(self):
        """Return (Approximate) Number of documents processed by current rank"""
    
    @abstractmethod
    def combine(self):
        """Combines individual rank files into one"""

    def to_jsonl(self, data: dict):
        """Utility function to convert dictionary to a json line. 
        
        Default converter is inefficient

        Returns:
            Json line in binary format
        """
        return json.dumps(data).encode("UTF-8") + b'\n'
    
    def count_lines(self, filename):
        """Utility function that counts number of documents in a jsonl file
        
        Refer to https://stackoverflow.com/a/850962 for more info on why it's efficient

        Args:
            filename (str): Path to a file
        
        Returns:
            (int) Number of Lines (documents) in the given jsonl file
        """
        f = open(filename)
        lines = 0
        buf_size = 1024 * 1024
        read_f = f.read # loop optimization

        buf = read_f(buf_size)
        while buf:
            lines += buf.count('\n')
            buf = read_f(buf_size)

        return lines
    
    



class LocalJsonlLoader(JsonlLoader):
    """Loads a jsonl file from local directory"""
    
    def load(self, load_path, save_dir=None):
        self.loader = jsonlines.open(load_path, mode='r')

        if save_dir is not None and save_dir != '':
            os.makedirs(save_dir, exist_ok = True)
            save_path = os.path.join(save_dir, f"{self.curr_rank}.jsonl")
            self.writer = open(save_path, mode = "wb") 

        # Note that this is an approximate length. 
        self.reader_length = self.count_lines(load_path) // self.world_size
        self.reader_length //= self.batch_size
        self.reader_length += 1 
    
    def __iter__(self):
        current_batch = []

        iterator = self.loader.iter(type=dict, skip_invalid=True, skip_empty=True)
        for idx, doc in enumerate(iterator):
            if (doc['text'] == ''):
                continue
            if not (idx % self.world_size == self.curr_rank):
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
    
    def close(self):
        self.loader.close()
        if hasattr(self, 'writer'):
            self.writer.close()
    
    def __len__(self):
        return self.reader_length

    def combine(self, save_dir):
        if self.curr_rank != 0:
            return
        
        save_path = os.path.join(save_dir, 'res.jsonl')
        save_fp = open(save_path, 'ab')

        for i in tqdm(range(0, self.world_size), desc = "Combining jsonl files"):
            buff_size = 1024*1024
            rank_path = os.path.join(save_dir, f'{i}.jsonl')
            with open(rank_path, 'rb') as rank_fp:
                while True:
                    buff = rank_fp.read(buff_size)
                    save_fp.write(buff)
                    if (len(buff) != buff_size):
                        break
            os.remove(rank_path)