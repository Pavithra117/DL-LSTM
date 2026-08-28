# DL- Developing a Deep Learning Model for NER using LSTM

## AIM
To develop an LSTM-based model for recognizing the named entities in the text.

## Problem Statement and Dataset


## DESIGN STEPS

STEP 1:
Load data, create word/tag mappings, and group sentences.

STEP 2:
Convert sentences to index sequences, pad to fixed length, and split into training/testing sets.

STEP 3:
Define dataset and DataLoader for batching.

STEP 4:
Build a bidirectional LSTM model for sequence tagging.

STEP 5:
Train the model over multiple epochs, tracking loss.

STEP 6:
Evaluate model accuracy, plot loss curves, and visualize predictions on a sample.




## PROGRAM

### Name: Pavithra K

### Register Number: 212224240112

```
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.nn.utils.rnn import pad_sequence

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



data = pd.read_csv("ner_dataset.csv", encoding="latin1").ffill()

print(data.head())


# Unique words and tags
words = list(data["Word"].unique())
tags = list(data["Tag"].unique())

# Add ENDPAD token
if "ENDPAD" not in words:
    words.append("ENDPAD")


# Create mappings
word2idx = {w: i + 1 for i, w in enumerate(words)}

tag2idx = {t: i for i, t in enumerate(tags)}

idx2tag = {i: t for t, i in tag2idx.items()}



print("Unique words in corpus:", data["Word"].nunique())
print("Unique tags in corpus:", data["Tag"].nunique())
print("Unique tags are:", tags)


print("\nEssential information about tagged entities:")
print("geo = Geographical Entity")
print("org = Organization")
print("per = Person")
print("gpe = Geopolitical Entity")
print("tim = Time indicator")
print("art = Artifact")
print("eve = Event")
print("nat = Natural Phenomenon")



class SentenceGetter:

    def __init__(self, data):

        self.grouped = data.groupby(
            "Sentence #",
            group_keys=False
        ).apply(
            lambda s: [
                (w, t)
                for w, t in zip(s["Word"], s["Tag"])
            ]
        )

        self.sentences = list(self.grouped)


getter = SentenceGetter(data)

sentences = getter.sentences

print("\nNumber of sentences:", len(sentences))

print("\nExample sentence:")
print(sentences[35])




X = [
    [word2idx[w] for w, t in sentence]
    for sentence in sentences
]

y = [
    [tag2idx[t] for w, t in sentence]
    for sentence in sentences
]


print("\nWord2idx sample:")
print(list(word2idx.items())[:10])



plt.hist(
    [len(s) for s in sentences],
    bins=50
)

plt.title("Sentence Length Distribution")
plt.xlabel("Sentence Length")
plt.ylabel("Frequency")
plt.show()




max_len = 50

X_pad = pad_sequence(
    [torch.tensor(seq) for seq in X],
    batch_first=True,
    padding_value=word2idx["ENDPAD"]
)

y_pad = pad_sequence(
    [torch.tensor(seq) for seq in y],
    batch_first=True,
    padding_value=tag2idx["O"]
)


X_pad = X_pad[:, :max_len]
y_pad = y_pad[:, :max_len]


print("\nX padded shape:", X_pad.shape)
print("Y padded shape:", y_pad.shape)


print("\nFirst X sequence:")
print(X_pad[0])

print("\nFirst Y sequence:")
print(y_pad[0])



X_train, X_test, y_train, y_test = train_test_split(
    X_pad,
    y_pad,
    test_size=0.2,
    random_state=1
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))



class NERDataset(Dataset):

    def __init__(self, X, y):

        self.X = X
        self.y = y

    def __len__(self):

        return len(self.X)

    def __getitem__(self, idx):

        return {
            "input_ids": self.X[idx],
            "labels": self.y[idx]
        }

train_dataset = NERDataset(
    X_train,
    y_train
)

test_dataset = NERDataset(
    X_test,
    y_test
)


train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)



class BiLSTMTagger(nn.Module):

    def __init__(
        self,
        vocab_size,
        tag_size,
        embedding_dim=100,
        hidden_dim=128
    ):

        super(BiLSTMTagger, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=word2idx["ENDPAD"]
        )

        # BiLSTM layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        # Fully connected layer
        self.fc = nn.Linear(
            hidden_dim * 2,
            tag_size
        )

    def forward(self, input_ids):

        # Word IDs -> embeddings
        embedded = self.embedding(input_ids)

        # BiLSTM
        lstm_output, _ = self.lstm(embedded)

        # Classification
        output = self.fc(lstm_output)

        return output



vocab_size = len(word2idx) + 1
tag_size = len(tag2idx)


model = BiLSTMTagger(
    vocab_size=vocab_size,
    tag_size=tag_size,
    embedding_dim=100,
    hidden_dim=128
).to(device)


print("\nModel:")
print(model)



loss_fn = nn.CrossEntropyLoss(
    ignore_index=tag2idx["O"]
)




optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)




def train_model(
    model,
    train_loader,
    test_loader,
    loss_fn,
    optimizer,
    epochs=3
):

    train_losses = []
    val_losses = []

    for epoch in range(epochs):


        model.train()

        total_train_loss = 0

        for batch in train_loader:

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids)

            outputs = outputs.view(
                -1,
                outputs.shape[-1]
            )

            labels = labels.view(-1)

            loss = loss_fn(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            total_train_loss += loss.item()


        avg_train_loss = (
            total_train_loss / len(train_loader)
        )


       

        model.eval()

        total_val_loss = 0

        with torch.no_grad():

            for batch in test_loader:

                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids)

                outputs = outputs.view(
                    -1,
                    outputs.shape[-1]
                )

                labels = labels.view(-1)

                loss = loss_fn(
                    outputs,
                    labels
                )

                total_val_loss += loss.item()


        avg_val_loss = (
            total_val_loss / len(test_loader)
        )


        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)


        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {avg_train_loss:.4f} "
            f"Val Loss: {avg_val_loss:.4f}"
        )


    return train_losses, val_losses



def evaluate_model(
    model,
    test_loader,
    X_test,
    y_test
):

    model.eval()

    true_tags = []
    pred_tags = []

    with torch.no_grad():

        for batch in test_loader:

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Prediction
            outputs = model(input_ids)

            preds = torch.argmax(
                outputs,
                dim=-1
            )


            for i in range(len(labels)):

                for j in range(len(labels[i])):

                    if labels[i][j].item() != tag2idx["O"]:

                        true_tags.append(
                            idx2tag[
                                labels[i][j].item()
                            ]
                        )

                        pred_tags.append(
                            idx2tag[
                                preds[i][j].item()
                            ]
                        )


    print("\nClassification Report:\n")

    print(
        classification_report(
            true_tags,
            pred_tags,
            zero_division=0
        )
    )


train_losses, val_losses = train_model(
    model,
    train_loader,
    test_loader,
    loss_fn,
    optimizer,
    epochs=3
)



evaluate_model(
    model,
    test_loader,
    X_test,
    y_test
)



print("\nName:Pavithra K  ")
print("Register Number: 212224240112")


history_df = pd.DataFrame({
    "loss": train_losses,
    "val_loss": val_losses
})


history_df.plot(
    title="Loss Over Epochs"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()


i = min(125, len(X_test) - 1)


model.eval()


sample = X_test[i].unsqueeze(0).to(device)


with torch.no_grad():

    output = model(sample)

    preds = torch.argmax(
        output,
        dim=-1
    ).squeeze().cpu().numpy()


true = y_test[i].numpy()



print("\nName: Pavithra K    ")
print("Register Number: 212224240112")

print(
    "{:<20} {:<10} {:<10}".format(
        "Word",
        "True",
        "Pred"
    )
)

print("-" * 45)


for w_id, true_tag, pred_tag in zip(
    X_test[i],
    y_test[i],
    preds
):

    if w_id.item() != word2idx["ENDPAD"]:

        word = words[w_id.item() - 1]

        true_label = tags[true_tag.item()]

        pred_label = tags[pred_tag]

        print(
            f"{word:<20} "
            f"{true_label:<10} "
            f"{pred_label:<10}"
        )
```

### OUTPUT

## Loss Vs Epoch Plot

<img width="862" height="636" alt="image" src="https://github.com/user-attachments/assets/28b85a01-83b4-4648-89c7-f17094c8e733" />


### Sample Text Prediction

<img width="531" height="498" alt="image" src="https://github.com/user-attachments/assets/5aba812a-5d60-48d1-8a76-eccdd31dc0ea" />


## RESULT
Thus, an LSTM-based model for recognizing the named entities in the text has been developed successfully.
