from modelwork import (clean_html, remove_spec_char, simple_stemmer,
                       remove_stop_words, load_model, load_vectorizer, predict)


def run_preprocessing_pipeline(text, functions) -> str:
    for function in functions:
        text = function(text)
    return text


def run_predict(text):
    preprocessing_functions = [
        clean_html,
        remove_spec_char,
        simple_stemmer,
        remove_stop_words,
    ]

    processed_text = run_preprocessing_pipeline(text, preprocessing_functions)

    model = load_model('models/mn_bays_tfidf_sentiment_analysis.pkl')
    vectorizer = load_vectorizer('models/vectorizer.pickle')

    result = predict(processed_text, model, vectorizer)

    print(result)


if __name__ == "__main__":
    input_text = "this show was garbage i am surprised it was approved. Next time the main character should do the right thing"
    run_predict(input_text)

