from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np

# Initialize the same embeddings model used in your vector store
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Test text
test_text = 'HuggingFace embedding test — no API key needed.'

# Get the embedding vector
vector = embeddings.embed_query(test_text)

print(f'Text: "{test_text}"')
print(f'Embedding dimension: {len(vector)}')
print(f'First 10 values: {vector[:10]}')
print(f'Value range: {min(vector):.4f} to {max(vector):.4f}')
print(f'Sample values:')
for i in range(0, len(vector), 50):  # Show every 50th value
    print(f'  [{i:3d}]: {vector[i]:.6f}')