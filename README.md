# sentiment-analysis-common

## Background
This is a common library that can be used to determine sentiment on various text. This was training on the 
[IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) originally this was
just practice on creating various models to learn. I wanted to expand it though, right now it primarily uses the 
Multinomial Naive Bayes model on top of a Term Frequency-Inverse Document Frequency (TFIDF) ranking. 

In the future there can be a more dynamic approach to selecting which model can be used. The Jupyter Notebook that is
training/creating the models created 4 unique models. 2 Linear Regression models and 2 Multinomial Naive Bayes models.
One model uses Term Frequency-Inverse Document Frequency (TFIDF) ranks while the other uses Bag of Words vectors.

I might make some API that interacts with this common library. I will also make a dockerfile eventually to have all this
setup happen.

[Hugging Face Link](https://huggingface.co/dsdevnull/mn_bays_tfidf_sentiment_analysis)

## Steps to setup
1. `pip install .` on the main directory
2. Run the Jupyter Notebook fully. That will create all the models that you need.
3. `python main.py` to run the code.
4. Test!
