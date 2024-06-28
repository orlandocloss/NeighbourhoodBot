# search-disconnection.py
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

def search_relevant_information(prompt, top_k=2, collection_name="disconnection"):
    embedding = get_embedding(prompt).tolist()
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

    if not utility.has_collection(collection_name):
        print(f"Collection {collection_name} does not exist.")
        return []

    collection = Collection(name=collection_name)
    collection.load()

    results = collection.search(
        data=[embedding], 
        anns_field="embeddings", 
        param=search_params, 
        limit=top_k, 
        expr=None,
        output_fields=["text"]
    )
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Search for relevant information in the disconnection collection")
    parser.add_argument('action', choices=['search'], help="Action to perform: 'search' for relevant information")
    parser.add_argument('text', type=str, help="Prompt to search for relevant information")
    args = parser.parse_args()

    if args.action == 'search':
        result = search_relevant_information(args.text, top_k=2)
        if result:
            top_sentences = [f'"{res.entity.get("text")}"' for res in result[0]]
            output = ", ".join(top_sentences)
            print(output)
            return output
        else:
            print("No relevant information found.")
            return "No relevant information found."

if __name__ == "__main__":
    main()
