import tarfile
import requests
import os


def download_recent_data():
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open("./data/aclImdb_v1.tar.gz", "wb") as f:
            f.write(response.content)


def read_tar_gz():
    file_path = "./data/aclImdb_v1.tar.gz"
    with tarfile.open(file_path, "r:gz") as tar:
        tar.extractall(path="./data/")
        print("extracted")

    # remove tar.gz file
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"{file_path} has been removed")
    else:
        raise FileNotFoundError(f"{file_path} does not exist")


if __name__ == "__main__":
    download_recent_data()
    read_tar_gz()
