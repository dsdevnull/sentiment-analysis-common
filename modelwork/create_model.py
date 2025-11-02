import tarfile
import requests
import os
import polars as pl
from pathlib import Path
from multiprocessing import Pool, cpu_count
from nltk.stem import PorterStemmer
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords

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


def load_text_file(base_dir: str) -> pl.Dataframe:
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


def preprocess_dataframe(raw_df: pl.Dataframe) -> pl.Dataframe:
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


def apply_stemmer_and_tokenizer(pre_df: pl.Dataframe) -> pl.Dataframe:
    print("stemming and removing stop words...")
    reviews = pre_df["review"].to_list()

    with Pool(cpu_count()) as p:
        normalized_reviews = p.map(stem_and_remove_stop_words, reviews)

    cleaned = pre_df.with_columns(pl.Series("review", normalized_reviews))
    return cleaned


if __name__ == "__main__":
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
    print(cleaned_df.head())
