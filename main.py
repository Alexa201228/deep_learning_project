import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from constants import TEST_DATA_PATH, BEST_MODEL_PATH

if __name__ == "__main__":
    test_df = pd.read_csv(TEST_DATA_PATH)
    random_test_text = test_df.sample(n=1)["article_content"].item()

    # TODO: this is not working! Find a way to load model and predict with torch
    model_path = BEST_MODEL_PATH
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    inputs = tokenizer(random_test_text, padding=True, truncation=True, return_tensors="pt")

    labels_for_prediction = {i: v for i, v in enumerate(test_df["category_tag"].unique().tolist(), start=0)}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    print(f"Text category is {labels_for_prediction[outputs]}")

