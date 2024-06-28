from pymilvus import utility, Collection, CollectionSchema, FieldSchema, DataType, connections

# Connect to Milvus database
connections.connect(alias='default', host='localhost', port='19530')

# Drop the existing collection if it exists
utility.drop_collection("problems")

# Define the problem_id field
problem_id = FieldSchema(
  name="problem_id",
  dtype=DataType.INT64,
  is_primary=True,
)

# Define the embeddings field
embeddings = FieldSchema(
  name="embeddings",
  dtype=DataType.FLOAT_VECTOR,
  dim=384,  # Specify the dimensionality of your embeddings
)

# Define the text field
text = FieldSchema(
  name="text",
  dtype=DataType.VARCHAR,
  max_length=65535  # Maximum length for text field
)

# Define the schema with the fields
schema = CollectionSchema(
  fields=[problem_id, embeddings, text],
  description="Collection for storing problems and their embeddings for similarity search"
)

# Collection name
collection_name = "problems"

# Create the collection
collection = Collection(
    name=collection_name,
    schema=schema,
    using='default',
    shards_num=2  # Adjust based on your scaling needs
)

def create_index_for_collection(collection_name, field_name="embeddings"):
    collection = Collection(name=collection_name)

    index_params = {
        "index_type": "IVF_FLAT",  # Example index type, choose based on your requirements
        "params": {"nlist": 64},  # Example params, adjust based on your vector field and search needs
        "metric_type": "L2"  # Example metric, choose L2 or IP based on your scenario
    }

    # Check if the index already exists to avoid recreating it unnecessarily
    if not collection.has_index():
        print(f"Creating index for collection '{collection_name}'...")
        collection.create_index(field_name=field_name, index_params=index_params)
        print("Index created.")
    else:
        print("Index already exists.")

# Call this function after inserting your data and before attempting to load the collection
create_index_for_collection("problems")

# Now, when you call collection.load() it should work as expected
