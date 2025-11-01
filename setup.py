import setuptools

setuptools.setup(
    name="sentiment-analysis-common",
    version="0.1.0",
    author="Duncan Squires",
    packages=setuptools.find_packages(),
    install_requires=[
        "matplotlib==3.9.0",
        "nltk==3.8.1",
        "numpy==2.0.0",
        "pandas==2.2.2",
        "scikit_learn==1.5.0",
        "seaborn==0.13.2",
        "spacy==3.7.5",
        "textblob==0.18.0.post0",
        "wordcloud==1.9.3",
        "huggingface_hub==0.23.4",
    ],
)
