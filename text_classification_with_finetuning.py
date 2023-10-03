# TODO: Continue fine-tuning experiment

import pandas as pd
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


def preprocess_function(examples):
    return tokenizer(examples["article_content"], truncation=True)


news_data = pd.read_csv("./data/train/scitechdaily.csv")

tokenized_news_data = news_data.applymap(preprocess_function)
