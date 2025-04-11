import numpy as np

def cosine_similarity(user_embedding, item_embeddings):
    if len(user_embedding):
        user_norm = np.linalg.norm(user_embedding)
        item_norms = np.linalg.norm(item_embeddings, axis=1)
        dot_product = np.dot(item_embeddings, user_embedding.flatten())
        similarities = dot_product / (user_norm * item_norms)
        return similarities
    else:
        return None


def find_topN_items(user_embedding, item_embeddings, topN):
    if len(user_embedding):
        similarities = cosine_similarity(user_embedding, item_embeddings)
        topN_indices = np.argsort(similarities)[-topN:][::-1]
        topN_similarities = similarities[topN_indices]
        return topN_similarities, topN_indices
    else:
        return None, None