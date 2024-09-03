# %%
from sentence_transformers import SentenceTransformer

# %%
model = SentenceTransformer("all-mpnet-base-v2")
# %%
docs: list[str] = [
    "My first paragraph. That contains information",
    "Python is a programming language.",
]
# %%
document_embeddings = model.encode(docs)
# %%
query = "What is Python?"
query_embedding = model.encode(query)
