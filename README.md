# Movie Recommendation System

A beginner-friendly Streamlit web application that recommends similar movies using a content-based recommendation approach.

## Features

- Modern dark-themed UI built with Streamlit
- Content-based recommendations using TF-IDF and cosine similarity
- Search box + movie dropdown + recommendation button
- Movie posters, genre, rating, and overview display
- Top Rated Movies section and genre filtering
- Loading spinner and error handling
- Sample dataset included in `movies.csv`

## Files

- `app.py` - Streamlit application code
- `movies.csv` - Sample movie dataset
- `requirements.txt` - Python dependencies
- `.streamlit/config.toml` - Streamlit theme configuration

## Prerequisites

- Python 3.10 or newer
- Internet access to load poster images

## Installation

1. Clone the repository or download the project files.
2. Open a terminal in the project folder.
3. Create a virtual environment (recommended):

```bash
python -m venv .venv
```

4. Activate the virtual environment:

- Windows:
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```

- macOS / Linux:
  ```bash
  source .venv/bin/activate
  ```

5. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run Locally

```bash
streamlit run app.py
```

Then open the displayed local URL in your browser.

## Streamlit Cloud Deployment

1. Push your repository to GitHub.
2. Visit [Streamlit Cloud](https://streamlit.cloud).
3. Connect your GitHub account and choose this repository.
4. Set the main file to `app.py`.
5. Deploy the app.

## GitHub Upload Steps

1. Initialize a Git repository:

```bash
git init
git add .
git commit -m "Initial movie recommendation system"
```

2. Create a GitHub repository and connect it:

```bash
git remote add origin https://github.com/<your-username>/<repo-name>.git
```

3. Push the code:

```bash
git push -u origin main
```

## Notes

- The sample dataset includes common movies with genres, plot descriptions, ratings, and poster links.
- You can expand `movies.csv` with your own movie titles to improve recommendations.
- If you want to use TMDB API directly, add a `poster_url` or `poster_path` field to the dataset and update the app accordingly.
