import torch
import pandas as pd
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from tools import dl_translator
from tools.constants import TEST_DATA_PATH, BEST_MODEL_PATH

tokenizer = get_tokenizer("basic_english")


def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text)


async def get_text_translation(file: bytes):

    """
    Method to translate user text from English to Russian
    :param file: text file for translation in bytes
    :return:
    """
    text = file.decode()
    translation = dl_translator.translate(text)
    return translation


async def get_recommendations_from_content(file) -> tuple[dict, str]:

    """
    Method to get recommendations for user based on uploaded file text
    :param file: file with text on which model makes recommendations
    :return: dict with articles' titles as keys and articles' links
    """
    fresh_news: pd.DataFrame = pd.read_csv(TEST_DATA_PATH)
    labels_for_prediction = {i: v for i, v in enumerate(fresh_news["category_tag"].unique().tolist(), start=0)}

    model = torch.jit.load(BEST_MODEL_PATH)
    model.eval()

    vocab = build_vocab_from_iterator(yield_tokens(file.decode()), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    text_pipeline = lambda x: vocab(tokenizer(x))

    with torch.no_grad():
        text = torch.tensor(text_pipeline(file.decode()))
        output = model(text, torch.tensor([0]))

    predicted_class = output.argmax(1).item() + 1

    recommendations_df: pd.DataFrame = fresh_news[fresh_news["category_tag"]
                                                  == labels_for_prediction[predicted_class]].sample(n=10)
    recommendations = {article["article_title"]: article["article_link"]
                       for _, article in recommendations_df.iterrows()}
    return recommendations, labels_for_prediction[predicted_class]




