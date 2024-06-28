import argparse
from sentence_transformers import SentenceTransformer
from pymilvus import Collection, connections, utility, DataType
import numpy as np

# Connect to Milvus database
connections.connect(alias='default', host='localhost', port='19530')

# Initialize the Sentence Transformer model
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
model = SentenceTransformer(model_name)

def get_embedding(text):
    return model.encode(text, convert_to_tensor=False)

def insert_data(problem_text):
    collection_name = "problems"
    
    if not utility.has_collection(collection_name):
        print(f"Collection {collection_name} does not exist.")
        return

    collection = Collection(name=collection_name)
    embedding = get_embedding(problem_text).tolist()
    
    # Get the next ID by checking the current count of entities
    current_count = collection.num_entities
    next_id = current_count + 1

    mr = collection.insert([
        [next_id],  # ID field
        [embedding],  # Embedding field
        [problem_text]  # Text field
    ])

    collection.flush()
    
    return mr.primary_keys

def search_similar_problems(prompt, top_k=5):
    embedding = get_embedding(prompt).tolist()
    collection_name = "problems"
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

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
    parser = argparse.ArgumentParser(description="Insert or Search Problems")
    parser.add_argument('action', choices=['insert', 'search'], help="Action to perform: 'insert' a new problem or 'search' for similar problems")
    parser.add_argument('text', type=str, help="Problem text to insert or search")
    args = parser.parse_args()

    if args.action == 'insert':
        result = insert_data(args.text)
        print(f"{result}")
        return str(result[0])
    elif args.action == 'search':
        result = search_similar_problems(args.text)
        entity = result[0][0]
        print(f"{entity.id}, {entity.distance}, {entity.entity.get('text')}")
        return (f"{entity.id}, {entity.distance}, {entity.entity.get('text')}")

if __name__ == "__main__":
    main()
