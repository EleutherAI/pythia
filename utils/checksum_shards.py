import os
import hashlib

# Path to the directory with files
directory_path = './pythia_deduped_pile_idxmaps'

# Read the shard_hashes.txt file
hashes_from_file = {}
with open('./utils/shard_hashes.txt', 'r') as file:
    for line in file:
        filename, sha256_hash = line.strip().split()
        hashes_from_file[filename] = sha256_hash

# Compare hashes
for filename in os.listdir(directory_path):
    file_path = os.path.join(directory_path, filename)
    if os.path.isfile(file_path):
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as file:
            while True:
                data = file.read(65536)  # Read the file in chunks
                if not data:
                    break
                sha256.update(data)
        
        file_hash = sha256.hexdigest()
        
        if filename in hashes_from_file:
            if file_hash == hashes_from_file[filename]:
                print(f"Match: {filename} - Hashes are the same")
            else:
                print(f"Mismatch: {filename} - Hashes differ")
        else:
            print(f"No hash found for file: {filename}")
