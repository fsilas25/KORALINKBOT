from bs4 import BeautifulSoup
from deep_translator import GoogleTranslator
import requests
import bot

gotr = GoogleTranslator(source='auto',target='en')

# --------------------------------------------------
# web scrapping
def scrape_page(url):
    response = requests.get(url)

    soup = BeautifulSoup(response.text,"html.parser")

    text = soup.get_text(separator=" ",strip=True)

    return text

#split long texts
def split_text(text, size=300):
    words = text.split()
    chunks = []
    for i in range(0, len(words), size):
        chunk = " ".join(
            words[i:i+size]
        )
        chunks.append(chunk)
    return chunks

website_text = scrape_page("https://koralink.org/en")

website_text = gotr.translate(website_text)

website_chunks = split_text(website_text)

website_embeddings = bot.embedding_model.encode(website_chunks,convert_to_numpy=True)

def search_website(query):
    query_embedding = bot.embedding_model.encode(
        query,
        convert_to_numpy=True
    )
    similarities = bot.cosine_similarity(
        [query_embedding],
        website_embeddings
    )[0]
    
    best_index = bot.np.argmax(similarities)
    print("Si:", best_index)
    
    return website_chunks[best_index]

# ----------------------------------------------------------------
# final answer retrieval method

def answer_finder(user_message):
    faq_answer = bot.answer_user_question(user_message)
    if faq_answer:
        return faq_answer
    return search_website(user_message)

