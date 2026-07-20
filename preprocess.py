import pandas as pd
import re
from nltk.corpus import stopwords

# had to run nltk.download('stopwords') once in the terminal before this
# worked, otherwise it throws a LookupError
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    import nltk
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)  # strip punctuation/special chars
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)


def load_data(path="movies.csv"):
    df = pd.read_csv(path)

    print("dataset shape:", df.shape)
    print("columns:", list(df.columns))
    print(df.head())

    # overview column has the actual plot description, thats the text we
    # want to compare between movies
    text_col = "overview"

    # some overviews were NaN which crashed the vectorizer the first time
    # i ran this, fillna fixes it
    df[text_col] = df[text_col].fillna("")
    if "genres" in df.columns:
        df["genres"] = df["genres"].fillna("")

    print("cleaning movie descriptions...")
    df["clean_text"] = df[text_col].apply(clean_text)

    # throwing the genre words in too so movies with the same genre get
    # pulled a little closer together, not just plot wording
    if "genres" in df.columns:
        df["clean_text"] = df["clean_text"] + " " + df["genres"].apply(clean_text)

    return df


if __name__ == "__main__":
    df = load_data()
    print("\nsample of cleaned text:")
    print(df[["title", "clean_text"]].head())
