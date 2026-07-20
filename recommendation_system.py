import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocess import load_data

df = load_data("movies.csv")

print("creating tf-idf matrix...")
tfidf = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
tfidf_matrix = tfidf.fit_transform(df["clean_text"])

print("tfidf matrix shape:", tfidf_matrix.shape)
print("vocab size:", len(tfidf.vocabulary_))

print("calculating similarity...")
# using cosine similarity instead of euclidean distance because it looks at
# the angle between two tfidf vectors, not the actual length of them. a
# short overview and a long overview that use similar words still end up
# close together, euclidean distance would just say theyre far apart
# because one vector is bigger
similarity = cosine_similarity(tfidf_matrix)
print("similarity matrix shape:", similarity.shape)

pickle.dump(similarity, open("similarity.pkl", "wb"))


def recommend(movie_name, top_n=5):
    titles_lower = df["title"].str.lower()
    movie_name = movie_name.lower()

    if movie_name not in titles_lower.values:
        print("movie not found in dataset:", movie_name)
        return []

    idx = titles_lower[titles_lower == movie_name].index[0]

    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    recs = []
    for i, score in scores:
        if i == idx:
            continue  # dont recommend the same movie to itself
        recs.append(df.iloc[i]["title"])
        if len(recs) == top_n:
            break

    return recs


if __name__ == "__main__":
    test_movies = ["The Dark Knight", "Titanic", "Toy Story"]

    for m in test_movies:
        print("\nmovies similar to", m)
        result = recommend(m, top_n=5)
        for r in result:
            print("-", r)
