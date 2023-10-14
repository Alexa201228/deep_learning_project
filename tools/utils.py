import nltk
import pandas as pd

from nltk.corpus import stopwords

from tools.constants import TEST_TEXT_PATH


def label_mapper(labels: list[str]):
    """
    Method to make a simple label encoding for pytorch
    :param labels: list of labels
    :return: dict with keys as labels (str) and their integers as values
    """
    labels_dict = {}
    for i in range(len(labels)):
        labels_dict[labels[i]] = i

    return labels_dict


def preprocess_data(path_to_data_csv: str):
    df = pd.read_csv(path_to_data_csv)
    df["article_content_preprocessed"] = df["article_content"].str.lower()
    df["article_title_preprocessed"] = df["article_title"].str.lower()

    # remove punctuations
    df["article_content_preprocessed"] = df["article_content_preprocessed"].str.replace("[^A-Za-z0-9]+", " ",
                                                                                        regex=True)
    df["article_title_preprocessed"] = df["article_title_preprocessed"].str.replace("[^A-Za-z0-9]+", " ", regex=True)

    # remove stopwords
    nltk.download("stopwords")

    stopwords_ = stopwords.words("english")

    df["article_content_preprocessed"] = df["article_content_preprocessed"].apply(
        lambda words: " ".join(word.lower() for word in words.split() if word not in stopwords_))
    df["article_title_preprocessed"] = df["article_title_preprocessed"].apply(
        lambda words: " ".join(word.lower() for word in words.split() if word not in stopwords_))

    df.to_csv(path_to_data_csv)


def save_test_text_to_txt(text: str):
    with open(TEST_TEXT_PATH, "w") as file:
        file.write(text)
