import os
import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
import re
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')


def download_dataset_to_df(url: str, file_path: str) -> pd.DataFrame:
    """
    Downloads dataset from kaggle directly into a dataframe.
    """
    try:
        return kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS, url, file_path)
    except ValueError:
        return kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS, url, file_path,
            pandas_kwargs={'encoding': "ISO-8859-1"}
        )


def clean_text(text):
    # tweet specific cleaning
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)  # remove urls
    text = re.sub(r"@\w+", "", text)  # remove mentions

    # general cleaning
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # remove special characters
    text = re.sub(r"\s+", " ", text).strip()  # remove extra white space

    # remove stop words
    stop_words = set(stopwords.words('english'))
    text = " ".join([word for word in text.split() if word not in stop_words])

    return text

