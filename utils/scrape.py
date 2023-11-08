import requests
import re

# This scrapes huggingface to get the pythia sha256 hashes because, so far as I can tell, there's no way to make huggingface deliver this information via their api
base_url = 'https://huggingface.co/datasets/EleutherAI/pythia_deduped_pile_idxmaps/blob/main/pile_0.87_deduped_text_document-'

sha256_list = []  # List to store extracted SHA256 hashes

with open("shard_hashes.txt", "w") as output_file:
    # Loop through the range of page numbers
    for i in range(83):  # Pages go from 00000 to 00082
        page_number = f"{i:05d}-of-00082.bin"  # Format the page number

        # Construct the full URL for the current page
        url = base_url + page_number
        file_name = url.split('/')[-1]

        # GET request to fetch the HTML content of the current page
        response = requests.get(url)

        if response.status_code == 200:
            html_content = response.text  # Get HTML content of the page

            # Regex to find SHA256 hash
            matches = re.findall(r'<strong>SHA256:</strong>\s*([\da-fA-F]+)', html_content)

            if matches:
                sha256_list.extend(matches)
                
                # Write filename and hash to the output file
                for sha256 in matches:
                    output_file.write(f"{file_name} {sha256}\n")
            else:
              print(f"No hash found for file: {file_name}")
        else:
            print(f"Failed to fetch HTML content from {url}. Status code: {response.status_code}")

# Print the collected SHA256 hashes
for idx, sha256 in enumerate(sha256_list, start=1):
    print(f"SHA256 hash {idx}: {sha256}")
