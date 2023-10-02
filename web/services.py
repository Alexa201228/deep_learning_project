import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import dl_translator


async def get_text_translation(file: bytes):

    """

    :param file:
    :return:
    """
    text = file.decode()
    print(text)
    translation = dl_translator.translate(text)
    print(translation)
    return translation


async def get_recommendations_from_content(file):

    """

    :param file:
    :return:
    """
    fresh_news = pd.read_csv("./../data/test/scitechdaily_test.csv")
    model_path = "./models/best_classification_model.pt"
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    inputs = tokenizer(file.decode(), padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    print(fresh_news.info())
    labels_for_prediction = {i: v for i, v in enumerate(fresh_news["category_tag"].unique().tolist(), start=0)}
    print(labels_for_prediction)
    print("This is a %s news" % labels_for_prediction[predicted_class])

    recommendations = {article["article_title"]: article["article_link"] for article
                       in fresh_news[fresh_news["category_tag"] == labels_for_prediction[predicted_class]].loc[:10, :]}

    return recommendations




