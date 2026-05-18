import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# App configuration
# -----------------------------
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply a little custom styling to make cards look cleaner
st.markdown(
    """
    <style>
    .movie-card { background-color: #111827; border: 1px solid #333; border-radius: 16px; padding: 18px; }
    .movie-card:hover { box-shadow: 0 6px 16px rgba(0, 0, 0, 0.35); }
    .movie-title { font-size: 1.12rem; font-weight: 700; margin-bottom: 0.4rem; }
    .movie-meta { color: #cbd5e1; margin-bottom: 0.75rem; }
    .movie-overview { color: #e2e8f0; font-size: 0.95rem; line-height: 1.5; }
    .sidebar .sidebar-content { background-color: #0f172a; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Helper functions
# -----------------------------
@st.cache_data
def load_data(csv_path: str) -> pd.DataFrame:
    """Load the movie dataset from a CSV file."""
    df = pd.read_csv(csv_path)
    df.fillna("", inplace=True)
    df["genre"] = df["genre"].astype(str)
    df["overview"] = df["overview"].astype(str)
    # Combine text fields for content-based similarity
    df["combined_features"] = (
        df["title"] + " " + df["genre"] + " " + df["overview"]
    )
    return df

@st.cache_data
def build_similarity_matrix(corpus: pd.Series):
    """Build TF-IDF and cosine similarity matrix for the movie corpus."""
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(corpus)
    return cosine_similarity(tfidf_matrix, tfidf_matrix)

def get_movie_index(df: pd.DataFrame, title: str) -> int:
    """Find the index of a movie title in the dataset."""
    try:
        return int(df[df["title"] == title].index[0])
    except IndexError:
        return -1

def recommend_movies(
    title: str,
    df: pd.DataFrame,
    similarity_matrix,
    top_n: int = 5,
) -> list[dict]:
    """Return a list of recommended movies given a selected movie."""
    movie_idx = get_movie_index(df, title)
    if movie_idx == -1:
        return []

    similarity_scores = list(enumerate(similarity_matrix[movie_idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = [score for score in similarity_scores if score[0] != movie_idx]
    top_indices = [idx for idx, _ in similarity_scores[:top_n]]
    return df.iloc[top_indices].to_dict(orient="records")

def display_movie_card(movie: dict):
    """Show a single movie recommendation card."""
    st.markdown(
        f"""
        <div class='movie-card'>
            <div style='display:flex; gap: 16px; flex-wrap: wrap;'>
                <img src='{movie.get('poster_url')}' alt='Poster' width='160' style='border-radius: 12px;' />
                <div style='max-width: 620px;'>
                    <div class='movie-title'>{movie.get('title')}</div>
                    <div class='movie-meta'>Genre: {movie.get('genre')} &#8226; Rating: {movie.get('rating')}</div>
                    <div class='movie-overview'>{movie.get('overview')}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------
# Data preparation
# -----------------------------
movies_df = load_data("movies.csv")
similarity_matrix = build_similarity_matrix(movies_df["combined_features"])

# Sidebar content
with st.sidebar:
    st.header("Movie Recommendation System")
    st.write(
        "A beginner-friendly content-based recommender that suggests similar movies using movie plot, genre, and ratings."
    )
    st.write("---")
    st.subheader("How to use")
    st.write(
        "1. Search or choose a movie from the dropdown.\n"
        "2. Click Recommend to find similar titles.\n"
        "3. Explore Top Rated Movies and genre filters."
    )
    st.write("---")
    st.subheader("Features")
    st.write(
        "- Content-based similarity with TF-IDF and cosine similarity.\n"
        "- Movie posters, ratings, and genre cards.\n"
        "- Top rated movies and genre filtering.\n"
        "- Dark theme and responsive layout."
    )
    st.write("---")
    st.caption("Built with Python, Streamlit, Pandas, and Scikit-learn.")

# Main page header
st.title("🎥 Movie Recommendation System")
st.markdown(
    "Find great movies similar to your favorite picks. Powered by content-based similarity with TF-IDF and cosine similarity."
)

# Search, selection and recommendation controls
recommendations = []
col1, col2 = st.columns([2, 1])
with col1:
    search_query = st.text_input("Search movies", placeholder="Type a movie title or keyword...")
    available_titles = movies_df["title"].tolist()
    if search_query:
        available_titles = [
            title
            for title in available_titles
            if search_query.lower() in title.lower()
        ]
    if not available_titles:
        st.warning("No movies match your search. Clear the search or try a different title.")
        available_titles = ["No movies found"]

    selected_movie = st.selectbox(
        "Select a movie", options=available_titles, index=0
    )

with col2:
    st.write("#")
    if st.button("Recommend"):
        if selected_movie == "No movies found":
            st.warning("Please choose a valid movie before clicking Recommend.")
        else:
            with st.spinner("Finding similar movies..."):
                recommendations = recommend_movies(selected_movie, movies_df, similarity_matrix, top_n=5)
            if recommendations:
                st.success(f"Top recommendations for '{selected_movie}'")
            else:
                st.error("No recommendations found. Try selecting another movie.")

# Recommendation display
if recommendations:
    st.subheader("Recommended Movies")
    for movie in recommendations:
        display_movie_card(movie)

# Genre filter and top rated section
st.write("---")
st.subheader("Top Rated Movies")

genre_options = ["All"] + sorted(movies_df["genre"].unique())
selected_genre = st.selectbox("Filter by genre", options=genre_options)

if selected_genre == "All":
    filtered_df = movies_df.copy()
else:
    filtered_df = movies_df[movies_df["genre"] == selected_genre]

top_rated = filtered_df.sort_values(by="rating", ascending=False).head(6)

if top_rated.empty:
    st.warning("No movies found for the selected genre.")
else:
    cols = st.columns(3)
    for idx, movie in enumerate(top_rated.to_dict(orient="records")):
        with cols[idx % 3]:
            st.image(movie.get("poster_url"), use_column_width=True)
            st.markdown(f"**{movie.get('title')}**")
            st.markdown(f"Genre: {movie.get('genre')}  ")
            st.markdown(f"Rating: {movie.get('rating')}  ")

# Show data sample for transparency
st.write("---")
with st.expander("See dataset sample"):
    st.dataframe(movies_df[["title", "genre", "rating"]].head(10))

# Footer and deployment note
st.markdown(
    "---\n"
    "**Tip:** Run this app locally with `streamlit run app.py`.\n"
    "For Streamlit Cloud, add this repository and deploy using the `app.py` entrypoint."
)
