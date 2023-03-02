import openai
from openai.embeddings_utils import get_embedding, cosine_similarity


openai.api_key = "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"

text = "We will combine the review summary and review text into a single combined text. The model will encode this combined text and output a single vector embedding."
model ="text-embedding-ada-002"
emb1 = get_embedding(text, engine=model)

emb2 = openai.Embedding.create(input = [text], model=model)
breakpoint()
