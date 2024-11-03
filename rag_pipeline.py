# Update the deprecated imports to the new ones
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load the scraped data
with open('metadata.txt', 'r', encoding='utf-8') as f:
    data = f.read()
print('Done loading scraped data....')
# Split the data into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_text(data)
print('Done splitting data to chunks....')
# Create embeddings using the Command R model from Ollama
embeddings = OllamaEmbeddings(model="command-r")
print('Done creating embeddings....')
# Create FAISS index from the chunks
vectorstore = FAISS.from_texts(chunks, embeddings)
print('Done creating FAISS index...')
# Save the FAISS index for later retrieval
vectorstore.save_local('faiss_index')
print('Saved FAISS index for future retrieval ...')