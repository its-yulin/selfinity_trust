# Assuming 'vector_store' is your Pinecone vector database instance
vector_store = Pinecone.from_existing_index(index_name=index_name,embedding=embeddings,text_key=text_field)
# Define your search query
# query_vector = [/* your query vector here */]

# Use the 'as_retriever' method to search for the top 8 most relevant entries
results = vector_store.as_retriever(search_kwargs={"k": 8}).search(query_vector)

# Extract and display the data entry including ID, values, and metadata
for result in results:
    entry_id = result.id
    values = result.values
    metadata = result.metadata
    print(f"ID: {entry_id}, Values: {values}, Metadata: {metadata}")
