# split-insert-knowledge-base.py
import re
import argparse
from sentence_transformers import SentenceTransformer
from pymilvus import Collection, connections, utility
import numpy as np

# Connect to Milvus database
connections.connect(alias='default', host='localhost', port='19530')

# Initialize the Sentence Transformer model
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
model = SentenceTransformer(model_name)

def get_embedding(text):
    return model.encode(text, convert_to_tensor=False)

def split_into_sentences(text):
    # Split text into sentences using regex
    sentence_endings = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s')
    return sentence_endings.split(text)

def insert_data(text, collection_name="disconnection"):
    if not utility.has_collection(collection_name):
        print(f"Collection {collection_name} does not exist.")
        return

    collection = Collection(name=collection_name)
    sentences = split_into_sentences(text)
    embeddings = [get_embedding(sentence).tolist() for sentence in sentences]
    
    # Get the next ID by checking the current count of entities
    current_count = collection.num_entities
    ids = list(range(current_count + 1, current_count + 1 + len(sentences)))

    mr = collection.insert([
        ids,  # ID field
        embeddings,  # Embedding field
        sentences  # Text field
    ])

    collection.flush()
    
    return mr.primary_keys

def main():
    parser = argparse.ArgumentParser(description="Split knowledge base into sentences and insert into Milvus")
    parser.add_argument('file_path', type=str, help="Path to the knowledge base file")
    args = parser.parse_args()

    # Read the knowledge base file
    with open(args.file_path, 'r') as file:
        knowledge_base_text = file.read()

    result = insert_data(knowledge_base_text)
    print(f"Inserted sentence IDs: {result}")

if __name__ == "__main__":
    main()
