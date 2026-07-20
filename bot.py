from sentence_transformers import SentenceTransformer
import numpy as np
import json

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

with open("data/koralink.json") as f:
    qa_pairs = json.load(f)

questions = [item["question"] for item in qa_pairs]
answers = [item["answer"] for item in qa_pairs]

question_embeddings = embedding_model.encode(questions)

FALLBACK_MESSAGE = (
    "I'm sorry, I couldn't find information about that. "
    "Insure the spellings. "
    "I'm designed to answer questions related to the KoraLink platform. "
    "You can ask me about jobs, agents, employers, the marketplace, wallet, payments, and other KoraLink features."
)


def _cosine_scores(query_embedding):
    return np.dot(question_embeddings, query_embedding) / (
        np.linalg.norm(question_embeddings, axis=1) * np.linalg.norm(query_embedding) + 1e-10
    )


def top_faq_candidates(user_question, top_k=5, min_score=0.85):
    """
    Returns up to top_k (question, answer, score) tuples, sorted by
    similarity, filtered to a low bar (min_score) so borderline matches
    still get passed along as candidates for the LLM to reason over.
    """
    q_emb = embedding_model.encode([user_question])[0]
    scores = _cosine_scores(q_emb)
    top_indices = np.argsort(scores)[::-1][:top_k]

    candidates = [
        (questions[i], answers[i], float(scores[i]))
        for i in top_indices
        if scores[i] >= min_score
    ]
    return candidates


def answer_user_question(user_question, threshold=0.45):
    """
    Kept for backward compatibility: a quick, direct FAQ lookup with no
    LLM involved. Still used as the fast-path for high-confidence hits.
    """
    candidates = top_faq_candidates(user_question, top_k=1, min_score=0.0)
    if not candidates:
        return FALLBACK_MESSAGE

    _, answer, score = candidates[0]
    print(f"Best Score: {score:.3f}")

    if score < threshold:
        return FALLBACK_MESSAGE
    return answer
