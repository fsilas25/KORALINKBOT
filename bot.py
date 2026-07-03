from sentence_transformers import SentenceTransformer
import numpy as np
import os
import json

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

with open("koralink_faqs_flat.json") as f:
    qa_pairs = json.load(f)

questions = [item["question"] for item in qa_pairs]
answers = [item["answer"] for item in qa_pairs]

question_embeddings = embedding_model.encode(questions)

def answer_user_question(user_question, threshold=0.55):
    q_emb = embedding_model.encode([user_question])[0]

    scores = np.dot(question_embeddings, q_emb) / (
        np.linalg.norm(question_embeddings, axis=1) * np.linalg.norm(q_emb)
    )
   

    best_index = np.argmax(scores)
    best_score = scores[best_index]

    print(f"Best Score: {best_score:.3f}")  

    if best_score < threshold:
        return (
            "I'm sorry, I couldn't find information about that. "
            "Insure the spellings. "
            " I'm designed to answer questions related to the KoraLink platform. "
            "You can ask me about jobs, agents, employers, the marketplace, wallet, payments, and other KoraLink features."
        )

    return answers[best_index]

