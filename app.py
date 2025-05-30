# book_recommender_app.py
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load dataset (update this if you're using Google Sheets instead)
@st.cache_data
def load_data():
    return pd.read_csv("books_dataset.csv")  # Replace with your actual path or Google Sheets logic

df = load_data()

# Preprocess book text into a single "searchable" string
df['search_text'] = df.apply(
    lambda row: f"{row['Title']} {row['Author(s)']} {row['Description']} {row['Genre']} {row['Fiction_or_Non_Fiction']}",
    axis=1
)

# Embed all books once
book_embeddings = model.encode(df['search_text'].tolist(), show_progress_bar=True)

# Streamlit UI
st.title("📚 AI Book Recommender")
query = st.text_input("What kind of books are you looking for today?", max_chars=300)

if query:
    query_embedding = model.encode([query])[0].reshape(1, -1)
    scores = cosine_similarity(query_embedding, book_embeddings)[0]
    top_indices = scores.argsort()[-5:][::-1]
    
    st.subheader("Top 5 Book Recommendations:")
    for idx in top_indices:
        book = df.iloc[idx]
        st.markdown(f"**{book['Title']}** by *{book['Author(s)']}*")
        st.markdown(f"⭐ {book['GoodReads_Rating']} | {book['Genre']} | {book['Year_of_Publishing']}")
        st.markdown(f"{book['Description'][:300]}...")
        if pd.notna(book['Our_Review']):
            st.markdown(f"[Read Our Review]({book['Our_Review']})")
        st.markdown("---")
