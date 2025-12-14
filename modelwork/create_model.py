import tarfile
from pathlib import Path
from typing import Tuple, List, Optional

import joblib
import polars as pl
import requests
from multiprocessing import Pool, cpu_count

from nltk.stem import PorterStemmer
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split


class SentimentClassifier:
    def __init__(
        self,
        data_dir: str = "./data",
        model_type: str = "log_r",  # "log_r" or "mnb"
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.model_type = model_type
        self.test_size = test_size
        self.random_state = random_state

        # NLTK tools
        self.tokenizer = TreebankWordTokenizer()
        self.stopword_list = set(stopwords.words("english"))
        self.ps = PorterStemmer()

        # Models
        self.bow_lr = LogisticRegression(
            penalty="l2", max_iter=500, C=1, random_state=42
        )
        self.tfidf_lr = LogisticRegression(
            penalty="l2", max_iter=500, C=1, random_state=42
        )
        self.bow_mnb = MultinomialNB()
        self.tfidf_mnb = MultinomialNB()

        # Vectorizers
        self.cv: Optional[CountVectorizer] = None
        self.tv: Optional[TfidfVectorizer] = None

        # Fitted models
        self.bow_model = None
        self.tfidf_model = None

    # ---------- Data acquisition ----------

    def download_recent_data(self) -> None:
        url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        out_path = self.data_dir / "aclImdb_v1.tar.gz"
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(out_path, "wb") as f:
            f.write(response.content)

    def extract_tar_gz(self, file_path: str) -> None:
        file_path = Path(file_path)
        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall(path=self.data_dir)
            print("extracted")

        if file_path.exists():
            file_path.unlink()
            print(f"{file_path} has been removed")
        else:
            raise FileNotFoundError(f"{file_path} does not exist")

    def load_text_file(self, base_dir: str) -> pl.DataFrame:
        base_dir = Path(base_dir)
        data: List[Tuple[str, str]] = []

        for label_dir in ["pos", "neg"]:
            dir_path = base_dir / label_dir
            label = "positive" if label_dir == "pos" else "negative"
            print("working...")
            for file_path in dir_path.glob("*.txt"):
                text = file_path.read_text(encoding="utf-8").strip()
                data.append((text, label))

        return pl.DataFrame(data, schema=["review", "sentiment"])

    # ---------- Preprocessing ----------

    def _preprocess_dataframe(self, raw_df: pl.DataFrame) -> pl.DataFrame:
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
        return cleaned

    def _stem_and_remove_stop_words(self, text: str) -> str:
        if not text:
            return ""
        tokens = self.tokenizer.tokenize(text)
        cleaned_tokens = [
            self.ps.stem(tok.lower())
            for tok in tokens
            if tok.lower() not in self.stopword_list and tok.isalnum()
        ]
        return " ".join(cleaned_tokens)

    def _apply_stemmer_and_tokenizer(self, pre_df: pl.DataFrame) -> pl.DataFrame:
        print("stemming and removing stop words...")
        reviews = pre_df["review"].to_list()
        with Pool(cpu_count()) as p:
            normalized_reviews = p.map(self._stem_and_remove_stop_words, reviews)

        cleaned = pre_df.with_columns(pl.Series("review", normalized_reviews))
        return cleaned

    # ---------- Splits & vectorizers ----------

    def _stratified_train_test_split(
        self,
        df: pl.DataFrame,
        text_col: str = "review",
        label_col: str = "sentiment",
    ) -> Tuple[List[str], List[str], List[str], List[str]]:
        x = df[text_col].to_list()
        y = df[label_col].to_list()

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=self.test_size, stratify=y, random_state=self.random_state
        )
        return x_train, x_test, y_train, y_test

    def _fit_vectorizers(
        self, x_train: List[str], x_test: List[str]
    ):
        # CountVectorizer
        self.cv = CountVectorizer(
            min_df=0.0, max_df=1.0, binary=False, ngram_range=(1, 2)
        )
        cv_train = self.cv.fit_transform(x_train)
        cv_test = self.cv.transform(x_test)
        print("BOW_cv_train:", cv_train.shape)
        print("BOW_cv_test:", cv_test.shape)

        # TfidfVectorizer
        self.tv = TfidfVectorizer(
            min_df=2, max_df=0.95, use_idf=True, ngram_range=(1, 2)
        )
        tv_train = self.tv.fit_transform(x_train)
        tv_test = self.tv.transform(x_test)
        print("Tfidf_train:", tv_train.shape)
        print("Tfidf_test:", tv_test.shape)

        return cv_train, cv_test, tv_train, tv_test

    # ---------- Training & evaluation ----------

    def _select_models(self):
        if self.model_type == "log_r":
            return self.bow_lr, self.tfidf_lr
        elif self.model_type == "mnb":
            return self.bow_mnb, self.tfidf_mnb
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def fit(self, df: pl.DataFrame) -> None:
        # full training pipeline given a pre-loaded raw dataframe
        cleaned = self._preprocess_dataframe(df)
        cleaned = self._apply_stemmer_and_tokenizer(cleaned)

        x_train, x_test, y_train, y_test = self._stratified_train_test_split(cleaned)
        cv_train, cv_test, tv_train, tv_test = self._fit_vectorizers(x_train, x_test)

        bow_base, tfidf_base = self._select_models()
        self.bow_model = bow_base.fit(cv_train, y_train)
        self.tfidf_model = tfidf_base.fit(tv_train, y_train)

        # store test sets for quick evaluation later if desired
        self._x_test = x_test
        self._y_test = y_test
        self._cv_test = cv_test
        self._tv_test = tv_test

    def evaluate(self) -> Tuple[float, float]:
        if any(
            v is None
            for v in [self.bow_model, self.tfidf_model, self._cv_test, self._tv_test]
        ):
            raise RuntimeError("Model must be fitted before evaluation.")

        bow_pred = self.bow_model.predict(self._cv_test)
        tfidf_pred = self.tfidf_model.predict(self._tv_test)

        bow_score = accuracy_score(self._y_test, bow_pred)
        tfidf_score = accuracy_score(self._y_test, tfidf_pred)
        return bow_score, tfidf_score

    # ---------- Inference & saving ----------

    def predict(self, texts: List[str], use_tfidf: bool = True) -> List[str]:
        if use_tfidf:
            if self.tv is None or self.tfidf_model is None:
                raise RuntimeError("TF-IDF model not trained.")
            X = self.tv.transform(texts)
            return self.tfidf_model.predict(X).tolist()
        else:
            if self.cv is None or self.bow_model is None:
                raise RuntimeError("BoW model not trained.")
            X = self.cv.transform(texts)
            return self.bow_model.predict(X).tolist()

    def save(self, model_path: str, vectorizer_path: str, use_tfidf: bool = True) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if use_tfidf:
            if self.tfidf_model is None or self.tv is None:
                raise RuntimeError("TF-IDF model/vectorizer not trained.")
            joblib.dump(self.tfidf_model, model_path)
            joblib.dump(self.tv, vectorizer_path)
        else:
            if self.bow_model is None or self.cv is None:
                raise RuntimeError("BoW model/vectorizer not trained.")
            joblib.dump(self.bow_model, model_path)
            joblib.dump(self.cv, vectorizer_path)


def load_or_download_imdb(data_dir: str = "./data") -> pl.DataFrame:
    data_dir_path = Path(data_dir)
    imdb_dir = data_dir_path / "aclImdb"
    output_csv = data_dir_path / "output.csv"

    clf = SentimentClassifier(data_dir=data_dir)

    if output_csv.exists():
        return pl.read_csv(output_csv)

    if imdb_dir.is_dir():
        df = clf.load_text_file(base_dir=str(imdb_dir / "train"))
    else:
        clf.download_recent_data()
        clf.extract_tar_gz(str(data_dir_path / "aclImdb_v1.tar.gz"))
        df = clf.load_text_file(base_dir=str(imdb_dir / "train"))

    df.write_csv(output_csv)
    return df


if __name__ == "__main__":
    development_flag = True
    data_dir = "./data"

    if development_flag:
        df = pl.read_csv(Path(data_dir) / "output.csv")
    else:
        df = load_or_download_imdb(data_dir=data_dir)

    clf = SentimentClassifier(
        data_dir=data_dir,
        model_type="log_r",
        test_size=0.2,
        random_state=42,
    )

    clf.fit(df)
    bow_score, tfidf_score = clf.evaluate()
    print(f"BOW accuracy: {bow_score:.4f}")
    print(f"TF-IDF accuracy: {tfidf_score:.4f}")

    clf.save("./models/model.pkl", "./models/vectorizer.pkl", use_tfidf=True)
