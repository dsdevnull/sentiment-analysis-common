import tarfile
from typing import Tuple

import requests
import os
import polars as pl
from pathlib import Path
from multiprocessing import Pool, cpu_count
from nltk.stem import PorterStemmer
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelBinarizer

# setting up new Stemmer and Tokenizer
tokenizer = TreebankWordTokenizer()
stopword_list = set(stopwords.words("english"))
ps = PorterStemmer()


def download_recent_data():
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open("./data/aclImdb_v1.tar.gz", "wb") as f:
            f.write(response.content)


def read_tar_gz(file_path: str):
    with tarfile.open(file_path, "r:gz") as tar:
        tar.extractall(path="./data/")
        print("extracted")

    # remove tar.gz file
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"{file_path} has been removed")
    else:
        raise FileNotFoundError(f"{file_path} does not exist")


def load_text_file(base_dir: str) -> pl.DataFrame:
    base_dir = Path(base_dir)
    data = []

    for label_dir in ["pos", "neg"]:
        dir_path = base_dir / label_dir
        label: str = "positive" if label_dir == "pos" else "negative"
        print("working...")
        for file_path in dir_path.glob("*.txt"):
            text = file_path.read_text(encoding="utf-8").strip()
            data.append((text, label))

    return pl.DataFrame(data, schema=["review", "sentiment"])


def preprocess_dataframe(raw_df: pl.DataFrame) -> pl.DataFrame:
    print("preprocessing...")
    cleaned = raw_df.with_columns(
        pl.col("review")
        .fill_null("")
        .str.replace_all(r"<[^>]*>", "")
        .str.replace_all(r"[^A-Za-z0-9\s]", "")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
        .str.to_lowercase()
        .alias("review")
    )
    print(cleaned.head())
    return cleaned


def stem_and_remove_stop_words(text: str) -> str:
    if not text:
        return ""
    tokens = tokenizer.tokenize(text)
    cleaned_tokens = [
        ps.stem(tok.lower())  # lowercase and stem
        for tok in tokens
        if tok.lower() not in stopword_list and tok.isalnum()
    ]
    return " ".join(cleaned_tokens)


def apply_stemmer_and_tokenizer(pre_df: pl.DataFrame) -> pl.DataFrame:
    print("stemming and removing stop words...")
    reviews = pre_df["review"].to_list()

    with Pool(cpu_count()) as p:
        normalized_reviews = p.map(stem_and_remove_stop_words, reviews)

    cleaned = pre_df.with_columns(pl.Series("review", normalized_reviews))
    return cleaned


def split_test_and_train_data(
    whole_df: pl.DataFrame, column_to_split: str = "review"
) -> Tuple[pl.Series, pl.Series]:
    rounded_num_rows = int(round(len(whole_df) * 0.8))
    train = whole_df[column_to_split][:rounded_num_rows]
    test = whole_df[column_to_split][rounded_num_rows:]
    return train, test


def count_vectorizer(
    train_df: pl.Series, test_df: pl.Series
) -> Tuple[pl.Series, pl.Series]:
    cv = CountVectorizer(min_df=0.0, max_df=1.0, binary=False, ngram_range=(1, 3))
    # transformed train reviews
    cv_train_reviews = cv.fit_transform(train_df)
    # transformed test reviews
    cv_test_reviews = cv.transform(test_df)

    print("BOW_cv_train:", cv_train_reviews.shape)
    print("BOW_cv_test:", cv_test_reviews.shape)
    return cv_train_reviews, cv_test_reviews


def term_freq_inverse_document_freq(
    train_df: pl.Series, test_df: pl.Series
) -> Tuple[pl.Series, pl.Series]:
    tv = TfidfVectorizer(min_df=0.0, max_df=1.0, use_idf=True, ngram_range=(1, 3))
    tv_train_reviews = tv.fit_transform(train_df)
    tv_test_reviews = tv.transform(test_df)
    print("Tfidf_train:", tv_train_reviews.shape)
    print("Tfidf_test:", tv_test_reviews.shape)
    return tv_train_reviews, tv_test_reviews


def label_binarizer(
    whole_df: pl.DataFrame,
) -> Tuple[pl.DataFrame, pl.Series, pl.Series]:
    lb = LabelBinarizer()
    sentiment = lb.fit_transform(whole_df["sentiment"])
    train_sentiment_data, test_sentiment_data = split_test_and_train_data(
        whole_df, column_to_split="sentiment"
    )
    return sentiment, train_sentiment_data, test_sentiment_data


def train_models(
    cv_train: pl.Series,
    tv_train: pl.Series,
    train_sentiment_data: pl.Series,
    model_type: str = "log_r",
) -> Tuple[pl.Model, pl.Model]:
    if model_type == "log_r":
        lr = LogisticRegression(penalty="l2", max_iter=500, C=1, random_state=42)
        bow_model = lr.fit(cv_train, train_sentiment_data)
        tfidf_model = lr.fit(tv_train, train_sentiment_data)
    elif model_type == "mnb":
        mnb = MultinomialNB()
        bow_model = mnb.fit(cv_train, train_sentiment_data)
        tfidf_model = mnb.fit(tv_train, train_sentiment_data)

    return bow_model, tfidf_model


if __name__ == "__main__":
    # Have this be a flag that can be sent from the api
    development_flag = True
    if development_flag:
        df = pl.read_csv("./data/output.csv")
    else:
        if os.path.isdir("./data/aclImdb"):
            df = load_text_file(base_dir="./data/aclImdb/train")
        else:
            download_recent_data()
            read_tar_gz()
            df = load_text_file(base_dir="./data/aclImdb/train")
        df.write_csv("./data/output.csv")

    cleaned_df = preprocess_dataframe(raw_df=df)
    cleaned_df = apply_stemmer_and_tokenizer(cleaned_df)
    train_data, test_data = split_test_and_train_data(cleaned_df)
    cv_train_data, cv_test_data = count_vectorizer(train_data, test_data)
    tv_train_data, tv_test_data = term_freq_inverse_document_freq(train_data, test_data)
    sentiment_data, train_sentiment, test_sentiment = label_binarizer(cleaned_df)
    multi_nb_bow, multi_nb_tfidf = train_models(
        cv_train_data, tv_train_data, train_sentiment
    )
