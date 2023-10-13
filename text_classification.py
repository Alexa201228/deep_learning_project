import torch
import pandas as pd
import time

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
from torchtext.data.functional import to_map_style_dataset

from classification_model_class import TextClassificationModel
from constants import TRAIN_DATA_PATH, TEST_DATA_PATH, BEST_MODEL_PATH
from utils import label_mapper, preprocess_data


# Select a device to run training and inference model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocess data
preprocess_data(TRAIN_DATA_PATH)
news_data = pd.read_csv(TRAIN_DATA_PATH)


def collate_batch(batch):
    """

    :param batch:
    :return:
    """
    label_list, text_list, offsets = [], [], [0]
    for _label, _text in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)


# Split to train and test datasets
train_data, test_data = train_test_split(news_data, test_size=0.2, random_state=42)


# Define a function to create the (label, text) tuple
def create_label_text_tuple(row):
    return row["category_tag"], row["article_content_preprocessed"]


# Apply the function to create the tuples for each row
train_data = train_data.apply(create_label_text_tuple, axis=1)
test_data = test_data.apply(create_label_text_tuple, axis=1)


# Load data with Dataloader
dataloader = DataLoader(
    train_data.tolist(), batch_size=16, shuffle=False, collate_fn=collate_batch
)

# Create tokenizer
tokenizer = get_tokenizer("basic_english")


def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)


# Create a vocabulary
vocab = build_vocab_from_iterator(yield_tokens(train_data), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

labels_dict = label_mapper(news_data["category_tag"].unique().tolist())

# Make pipelines for text classification
text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: labels_dict.get(x, -1)

# Create model
num_class = len(news_data["category_tag"].unique())
vocab_size = len(vocab)
emsize = 64
model = TextClassificationModel(vocab_size, emsize, num_class).to(device)


def train(dataloader):
    """

    :param dataloader:
    :return:
    """
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500

    for idx, (label, text, offsets) in enumerate(dataloader):
        optimizer.zero_grad()
        predicted_label = model(text, offsets)
        loss = criterion(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            print(
                "| epoch {:3d} | {:5d}/{:5d} batches "
                "| accuracy {:8.3f}".format(
                    epoch, idx, len(dataloader), total_acc / total_count
                )
            )
            total_acc, total_count = 0, 0


def evaluate(dataloader):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            predicted_label = model(text, offsets)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc / total_count


# Hyperparameters
EPOCHS = 30  # epoch
LR = 5  # learning rate
BATCH_SIZE = 32  # batch size for training

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.1)
total_accu = None
train_dataset = to_map_style_dataset(train_data)
test_dataset = to_map_style_dataset(test_data)
num_train = int(len(train_dataset) * 0.95)
split_train_, split_valid_ = random_split(
    train_dataset, [num_train, len(train_dataset) - num_train]
)

train_dataloader = DataLoader(
    split_train_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch
)
valid_dataloader = DataLoader(
    split_valid_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch
)
test_dataloader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch
)

prev_accuracy, train_accuracy = 0.0, 0.0

for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    train(train_dataloader)
    accu_val = evaluate(valid_dataloader)
    if total_accu is not None and total_accu > accu_val:
        scheduler.step()
    else:
        total_accu = accu_val
    print("-" * 59)
    print(
        "| end of epoch {:3d} | time: {:5.2f}s | "
        "valid accuracy {:8.3f} ".format(
            epoch, time.time() - epoch_start_time, accu_val
        )
    )
    train_accuracy = accu_val
    if train_accuracy > prev_accuracy:
        print(f"Found better model! With {accu_val=}")
        model_scripted = torch.jit.script(model)  # Export to TorchScript
        model_scripted.save(BEST_MODEL_PATH)  # Save
        prev_accuracy = train_accuracy

    print("-" * 59)


print("Checking the results of test dataset.")
accu_test = evaluate(test_dataloader)
print("test accuracy {:8.3f}".format(accu_test))


def predict(text, text_pipeline):
    with torch.no_grad():
        text = torch.tensor(text_pipeline(text))
        output = model(text, torch.tensor([0]))
        return output.argmax(1).item() + 1


preprocess_data(TEST_DATA_PATH)
actual_test_data = pd.read_csv(TEST_DATA_PATH)

sample = actual_test_data.sample(n=1)
ex_text_str = sample["article_content_preprocessed"].item()
actual_label = sample["category_tag"].item()

labels_for_prediction = {i: v for i, v in enumerate(actual_test_data["category_tag"].unique().tolist(), start=1)}

model = torch.jit.load(BEST_MODEL_PATH)
model.eval()

print(f"{labels_for_prediction}")
print("This is a %s news" % labels_for_prediction[predict(ex_text_str, text_pipeline)])
print(f"{actual_label=}")
print(f"{sample['article_title'].item()=}")
