import chromadb
from functions import readtextfiles, chunksplitter, getembedding

chromaclient = chromadb.HttpClient(host="localhost", port=8000)
textdocspath = "files"
text_data = readtextfiles(textdocspath)

# Check for existing collections
existing_collections = [col.name for col in chromaclient.list_collections()]
if "buildragwithpython" in existing_collections:
    chromaclient.delete_collection("buildragwithpython")

# Recreate the collection
collection = chromaclient.get_or_create_collection(name="buildragwithpython", metadata={"hnsw:space": "cosine"})

# Process and add documents
for filename, text in text_data.items():
    chunks = chunksplitter(text)
    embeds = getembedding(chunks)  # Ensure embeddings are generated successfully
    chunknumber = list(range(len(chunks)))
    ids = [filename + str(index) for index in chunknumber]
    metadatas = [{"source": filename} for index in chunknumber]
    collection.add(ids=ids, documents=chunks, embeddings=embeds, metadatas=metadatas)