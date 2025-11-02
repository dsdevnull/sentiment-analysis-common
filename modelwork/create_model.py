import tarfile
import requests
import os
import polars as pl
from pathlib import Path


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



if __name__ == "__main__":
    if os.path.isdir("./data/aclImdb"):
        df = load_text_file(base_dir="./data/aclImdb/train")
    else:
        download_recent_data()
        read_tar_gz()
        df = load_text_file(base_dir="./data/aclImdb/train")
    print(df.head())

    #continue training below...
    #write out csv before model creation
