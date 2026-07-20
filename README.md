# Assignment 20 - Movie Recommendation System

Dataset: TMDB 5000 Movies Dataset (title, overview, genres columns)
Kaggle link: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

I picked this one because the overview column is basically a short plot
summary for every movie, which is perfect for TF-IDF - similar plots should
use similar words. genres was a nice bonus column to throw in too. Note:
the full TMDB file has like 5000 rows and a bunch of columns I don't need
(budget, revenue, production companies etc), so movies.csv here is a
trimmed down version I made with ~47 movies and just title/overview/genres,
enough to actually test that the recommender works without loading a giant
csv every time I re-ran the script. Swap it for the real Kaggle csv
(filtered to those 3 columns) if you want recommendations across a much
bigger movie list.

## Libraries used
pandas, numpy, scikit-learn (TfidfVectorizer + cosine_similarity), nltk
(stopwords), streamlit

## Files
- preprocess.py - loads movies.csv, fills NaN overviews, cleans the text
  (lowercase, strip punctuation, remove stopwords)
- recommendation_system.py - builds the TF-IDF matrix, computes cosine
  similarity, saves similarity.pkl, has the recommend() function
- app.py - streamlit UI, dropdown + button
- movies.csv - the trimmed dataset
- similarity.pkl - generated automatically when you run
  recommendation_system.py (already included here too)

## How to run
pip install -r requirements.txt
python recommendation_system.py   (this regenerates similarity.pkl and prints test recommendations)
streamlit run app.py

## Steps I did
1. loaded movies.csv with pandas, printed shape/columns/head to see what I
   was working with
2. overview had some NaN values (a few movies didn't have descriptions),
   replaced with fillna("")
3. wrote a clean_text function - lowercase, regex to strip punctuation,
   remove stopwords using nltk. also appended the genre words onto the
   clean text so genre would count too, not just plot wording
4. TfidfVectorizer with max_features=5000, stop_words='english',
   ngram_range=(1,2) so it picks up two-word phrases too, not just single
   words
5. cosine_similarity on the tfidf matrix, saved it with pickle so app.py
   doesn't need to rebuild it every single time (though right now it kind
   of does anyway since app.py imports recommendation_system.py directly,
   see "problems I faced" below)
6. recommend() function - finds the movie's row index, sorts the
   similarity scores for that row descending, skips the movie itself,
   returns the top N titles

## Sample recommendation
movies similar to The Dark Knight
- Batman Begins
- The Godfather
- Prisoners
- Se7en
- Guardians of the Galaxy

(makes sense, Batman Begins is the obvious one since its literally the
same character, and Godfather/Prisoners/Se7en are all crime dramas so the
overview wording overlaps a lot with Dark Knight's crime/Gotham themes)

## Problems I faced
- got a ValueError from TfidfVectorizer the first time I ran it because a
  couple of overview cells were NaN (float, not string). fillna("") on the
  overview column before cleaning fixed it.
- I first tried CountVectorizer just to get something working, but the
  recommendations were noticeably worse - common words like "man" or "life"
  showing up in a lot of overviews were dragging unrelated movies together.
  switching to TfidfVectorizer fixed most of that since it down-weights
  words that show up everywhere.
- app.py does `from recommendation_system import df, recommend` which
  means every time streamlit reruns the script it re-loads the csv,
  re-cleans the text and rebuilds the whole tfidf matrix from scratch
  instead of just loading similarity.pkl. works fine for 47 rows but I
  know this isn't really how you're supposed to use the pickle file, ran
  out of time to fix it properly with the full dataset.
- had to run nltk.download('stopwords') once in a python shell before the
  stopwords import would work, kept getting a LookupError until I did that.

## What I learned
Learned how TF-IDF actually differs from just counting words (TF-IDF cares
about how rare/common a word is across ALL the documents, not just how
often it shows up in one). Also learned why cosine similarity makes more
sense than something like euclidean distance for text vectors - it only
cares about the angle between two vectors so a short overview and a long
one can still score high if they're about similar stuff. ngram_range=(1,2)
was new to me too, before this I only ever used single words (unigrams).

## Future improvements
- use the actual full TMDB 5000 csv instead of my trimmed 47 movie version
- add movie posters using the TMDB API so the UI looks nicer
- fix app.py to actually load similarity.pkl instead of rebuilding
  everything on every rerun
- maybe try combining this with a bit of collaborative filtering later on
  if we cover that in a future assignment

## GitHub steps
git init
git add .
git commit -m "movie recommendation system"
git branch -M main
git remote add origin <your-repo-url-here>
git push -u origin main

## Render deployment steps
1. push this repo to GitHub first (steps above)
2. go to render.com, New -> Web Service, connect the GitHub repo
3. Build command: pip install -r requirements.txt
4. Start command: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
5. runtime: Python 3 (render usually auto detects this from requirements.txt)
6. once its deployed Render gives you a url, something like
   Replace with your deployed Render URL after deployment.
