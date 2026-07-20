
from bs4 import BeautifulSoup
from deep_translator import GoogleTranslator
import requests
import numpy as np
import bot
import ollama_client
import pdf_source

gotr = GoogleTranslator(source='auto', target='en')

# --------------------------------------------------
# web scrapping
def scrape_page(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    return text


# split long texts
def split_text(text, size=300):
    words = text.split()
    chunks = []
    for i in range(0, len(words), size):
        chunk = " ".join(words[i:i + size])
        chunks.append(chunk)
    return chunks


website_text = scrape_page("https://koralink.org/en")
website_text = gotr.translate(website_text)
website_chunks = split_text(website_text)
website_embeddings = bot.embedding_model.encode(website_chunks, convert_to_numpy=True)


def _cosine_sim(query_embedding, corpus_embeddings):
    return np.dot(corpus_embeddings, query_embedding) / (
        np.linalg.norm(corpus_embeddings, axis=1) * np.linalg.norm(query_embedding) + 1e-10
    )


def top_website_chunks(query, top_k=3, min_score=0.2):
    """Returns up to top_k (chunk_text, score) tuples from the scraped website."""
    query_embedding = bot.embedding_model.encode(query, convert_to_numpy=True)
    similarities = _cosine_sim(query_embedding, website_embeddings)
    top_indices = np.argsort(similarities)[::-1][:top_k]

    return [
        (website_chunks[i], float(similarities[i]))
        for i in top_indices
        if similarities[i] >= min_score
    ]


def search_website(query):
    """Kept for backward compatibility - single best chunk, text only."""
    chunks = top_website_chunks(query, top_k=1, min_score=0.0)
    return chunks[0][0] if chunks else ""


# High-confidence bar: if the FAQ match is this good, skip the LLM entirely
# (fast + free). Anything below this pools every source together instead.
FAQ_HIGH_CONFIDENCE = 0.55

# How many of the best candidates (across ALL sources combined) get sent
# to the LLM as context.
MAX_CANDIDATES_FOR_LLM = 6


def retrieve_all_candidates(user_message):
    """
    Gathers candidates from every source - FAQ JSON, scraped website,
    and PDF - each tagged with its source and similarity score, then
    sorts the pooled list so the strongest evidence (whichever source it
    came from) sits at the top.

    Returns (ranked_candidates, faq_candidates) - faq_candidates is
    returned separately too, since it's used for the fast-path check.
    """
    candidates = []

    faq_candidates = bot.top_faq_candidates(user_message, top_k=5, min_score=0.25)
    for question, answer, score in faq_candidates:
        candidates.append(("FAQ", answer, score))

    website_candidates = top_website_chunks(user_message, top_k=3, min_score=0.2)
    for chunk, score in website_candidates:
        candidates.append(("Website", chunk, score))

    pdf_candidates = pdf_source.top_pdf_chunks(user_message, top_k=3, min_score=0.2)
    for chunk, score in pdf_candidates:
        candidates.append(("PDF", chunk, score))

    candidates.sort(key=lambda c: c[2], reverse=True)
    return candidates, faq_candidates


# ----------------------------------------------------------------
# final answer retrieval method
def answer_finder(user_message):
    ranked_candidates, faq_candidates = retrieve_all_candidates(user_message)

    # Fast path: a strong direct FAQ match, no need to call the LLM.
    if faq_candidates and faq_candidates[0][2] >= FAQ_HIGH_CONFIDENCE:
        return faq_candidates[0][1]

    # Otherwise, hand the LLM the best-scoring evidence pooled from every
    # source (FAQ + website + PDF) - it decides which one(s) actually
    # answer the question and responds using only that.
    top_candidates = ranked_candidates[:MAX_CANDIDATES_FOR_LLM]

    llm_answer = ollama_client.get_llm_answer(user_message, top_candidates)
    if llm_answer:
        return llm_answer

    return bot.FALLBACK_MESSAGE
