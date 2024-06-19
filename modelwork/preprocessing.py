import re

import nltk
from nltk.stem.porter import PorterStemmer
from nltk.tokenize.toktok import ToktokTokenizer


def clean_html(line) -> str:
    clean_data = re.compile("<.*?>")
    return re.sub(clean_data, "", line)


def remove_spec_char(line) -> str:
    pattern = r"[^a-zA-z0-9\s]"
    return re.sub(pattern, "", line)


def simple_stemmer(text):
    ps = PorterStemmer()
    text = " ".join([ps.stem(word) for word in text.split()])
    return text


def remove_stop_words(line, is_lower_case=False) -> str:
    tokenizer = ToktokTokenizer()
    stopword_list = nltk.corpus.stopwords.words("english")
    tokens = tokenizer.tokenize(line)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [
            token for token in tokens if token.lower() not in stopword_list
        ]
    filtered_text = " ".join(filtered_tokens)
    return filtered_text
