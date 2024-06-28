from pymilvus import utility, connections

# Connect to Milvus database
connections.connect(alias='default', host='localhost', port='19530')

# Define the collection name
collection_name = "problems"

# Check if the collection exists
if utility.has_collection(collection_name):
    # Drop the collection
    utility.drop_collection(collection_name)
    print(f"Collection '{collection_name}' has been deleted.")
else:
    print(f"Collection '{collection_name}' does not exist.")
