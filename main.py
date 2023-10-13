import pandas as pd
import torch
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


from constants import TEST_DATA_PATH, BEST_MODEL_PATH

if __name__ == "__main__":
    test_df = pd.read_csv(TEST_DATA_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random_test_text = test_df.sample(n=1)["article_content"].item()

    labels_for_prediction = {i: v for i, v in enumerate(test_df["category_tag"].unique().tolist(), start=0)}
    tokenizer = get_tokenizer("basic_english")

    model = torch.jit.load(f"./.{BEST_MODEL_PATH}")
    model.eval()

    def yield_tokens(data_iter):
        for text in data_iter:
            yield tokenizer(text)


    vocab = build_vocab_from_iterator(yield_tokens(random_test_text), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    text_pipeline = lambda x: vocab(tokenizer(x))

    with torch.no_grad():
        text = torch.tensor(text_pipeline(random_test_text))
        output = model(text, torch.tensor([0]))

    predicted_class = output.argmax(1).item() + 1

    print(f"Text category is {labels_for_prediction[predicted_class]}")
