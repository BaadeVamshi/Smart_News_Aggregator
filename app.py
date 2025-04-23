from flask import Flask, render_template, url_for
from newspaper import Article, build
import nltk
from nltk.tokenize import word_tokenize
import requests
from time import sleep
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from transformers import pipeline

# Load BART summarization pipeline once
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

DEFAULT_IMAGES = {
    "https://indianexpress.com/": "https://upload.wikimedia.org/wikipedia/en/5/56/Indian_Express_logo.png",
    "https://timesofindia.indiatimes.com/": "https://th.bing.com/th/id/OIP.R5_weenE-s75MDvWzWte2wHaCE?w=349&h=97&c=8&rs=1&qlt=90&r=0&o=6&dpr=1.3&pid=3.1&rm=2",  # ✅ Default Times of India image
    "https://www.deccanchronicle.com/": "https://www.deccanchronicle.com/images/logo.jpg"
}

# Ensure nltk 'punkt' is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Disable SSL certificate verification (not recommended for production)
requests.adapters.DEFAULT_RETRIES = 5
session = requests.Session()
adapter = requests.adapters.HTTPAdapter(max_retries=3)
session.mount("https://", adapter)
Article.session = session

app = Flask(__name__)

def summarize_with_bart(text):
    try:
        if len(text.split()) > 1000:
            text = " ".join(text.split()[:1000])  # Truncate very long articles
        summary = summarizer(text, max_length=130, min_length=80, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        return "Summary not available due to error."
def split_text(text, chunk_size=400):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# chunks = split_text(result["text"], chunk_size=400) 
# for chunk in chunks:
#     print(chunk+"\n")
# chunk_summaries = []
# for i, chunk in enumerate(chunks):
#     print(f"Summarizing chunk {i+1}/{len(chunks)}")
#     summary = summarizer(chunk, max_length=100, min_length=50, do_sample=False)[0]["summary_text"]
#     chunk_summaries.append(summary)

# final_text = " ".join(chunk_summaries)

# print("\n Final Summary:\n", final_text)

def fetch_articles(url, max_count):
    paper = build(url, memoize_articles=False)
    articles = []
    retries = 3
    for article in paper.articles:
        attempt = 0
        while attempt < retries:
            try:
                article.download()
                article.parse()
                article.nlp()
                if len(article.text.split()) > 50:
                    image = article.top_image if article.top_image else DEFAULT_IMAGES.get(url, "")
                    articles.append({
                        "title": article.title,
                        "text": article.text,
                        "image": image,
                        "summary": summarize_with_bart(article.text),
                        "source": url,
                        "link": article.url 
                    })
                if len(articles) >= max_count:
                    break
                break
            except Exception as e:
                attempt += 1
                sleep(2)
                if attempt == retries:
                    continue
    return articles


# Fetch articles
indian_express_articles = fetch_articles("https://indianexpress.com/", max_count=8)
times_of_india_articles = fetch_articles("https://timesofindia.indiatimes.com/", max_count=10)
deccan_chronicle_articles = fetch_articles("https://www.deccanchronicle.com/", max_count=10)

all_articles = indian_express_articles + times_of_india_articles + deccan_chronicle_articles

@app.route('/')
def index():
    return render_template('index.html',
        newspapers=[
            {
                "name": "Indian Express",
                "id": "indian-express",
                "image": indian_express_articles[0]['image'] if indian_express_articles else DEFAULT_IMAGES["https://indianexpress.com/"]
            },
            {
                "name": "Times of India",
                "id": "times-of-india",
                "image": times_of_india_articles[0]['image'] if times_of_india_articles else DEFAULT_IMAGES["https://timesofindia.indiatimes.com/"]
            },
            {
                "name": "Deccan Chronicle",
                "id": "deccan-chronicle",
                "image": deccan_chronicle_articles[0]['image'] if deccan_chronicle_articles else DEFAULT_IMAGES["https://www.deccanchronicle.com/"]
            }
        ])

@app.route('/articles/<paper_id>')
def show_articles(paper_id):
    if paper_id == "indian-express":
        articles = indian_express_articles
        name = "Indian Express"
    elif paper_id == "times-of-india":
        articles = times_of_india_articles
        name = "Times of India"
    elif paper_id == "deccan-chronicle":
        articles = deccan_chronicle_articles
        name = "Deccan Chronicle"
    else:
        articles = []
        name = "Unknown"
    return render_template('articles.html', articles=articles, newspaper_name=name)

@app.route('/common_news')
def common_news():
    # Tag source in each article for grouping
    for article in indian_express_articles:
        article['source'] = 'Indian Express'
    for article in times_of_india_articles:
        article['source'] = 'Times of India'
    for article in deccan_chronicle_articles:
        article['source'] = 'Deccan Chronicle'

    all_articles = indian_express_articles + times_of_india_articles + deccan_chronicle_articles

    combined_texts = [(a['title'] + " " + a['summary']) for a in all_articles]

    tfidf = TfidfVectorizer(stop_words='english').fit_transform(combined_texts)
    similarity_matrix = cosine_similarity(tfidf)

    threshold = 0.20  
    common_news_groups = []

    for i in range(len(all_articles)):
        sources_found = {all_articles[i]['source']}
        group = [all_articles[i]]
        for j in range(len(all_articles)):
            if i != j and similarity_matrix[i][j] > threshold:
                if all_articles[j]['source'] not in sources_found:
                    group.append(all_articles[j])
                    sources_found.add(all_articles[j]['source'])
        if len(sources_found) == 3:
            # Avoid duplicates
            already_exists = any(set(a['title'] for a in group).issubset(set(b['title'] for b in g)) for g in common_news_groups)
            if not already_exists:
                common_news_groups.append(group)

    return render_template('common_news.html', groups=common_news_groups)

if __name__ == '__main__':
    app.run(debug=True)
    
    
    
    
    
    
    
