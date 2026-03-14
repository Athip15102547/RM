import torch
import pandas as pd
import random
import numpy as np

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from dataset import SpamDataset
from model import SpamModel

from tqdm import tqdm


# =============================
# Seed for reproducibility
# =============================
def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed()


# =============================
# Device
# =============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("CUDA:", torch.cuda.is_available())

def main():

    set_seed()

    print("CUDA:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))


# =============================
# Load Dataset
# =============================
    df = pd.read_csv("dataset/sms_spam.csv")

    print("Dataset size:", len(df))


# =============================
# Train / Validation split
# =============================
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df["text"],
        df["label"],
        test_size=0.1,
        stratify=df["label"],
        random_state=42
    )


# =============================
# Dataset
# =============================
    train_dataset = SpamDataset(
        train_texts.tolist(),
        train_labels.tolist()
    )

    val_dataset = SpamDataset(
        val_texts.tolist(),
        val_labels.tolist()
    )


    # =============================
    # DataLoader
    # =============================
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        num_workers=2
    )


    # =============================
    # Model
    # =============================

    model = SpamModel().to(device)


    # =============================
    # Optimizer / Loss
    # =============================
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-5
    )

    criterion = torch.nn.CrossEntropyLoss()

    epochs = 3


    # =============================
    # Evaluation
    # =============================
    def evaluate(model, dataloader):

        model.eval()

        preds = []
        labels = []

        total_loss = 0

        with torch.no_grad():

            for batch in dataloader:

                input_ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                y = batch["label"].to(device)

                outputs = model(input_ids, mask)

                loss = criterion(outputs, y)

                total_loss += loss.item()

                p = torch.argmax(outputs, dim=1)

                preds.extend(p.cpu().numpy())
                labels.extend(y.cpu().numpy())

        acc = accuracy_score(labels, preds)
        precision = precision_score(labels, preds)
        recall = recall_score(labels, preds)
        f1 = f1_score(labels, preds)

        avg_loss = total_loss / len(dataloader)

        return acc, precision, recall, f1, avg_loss


    # =============================
    # Training
    # =============================
    best_f1 = 0

    for epoch in range(epochs):

        model.train()

        loop = tqdm(train_loader)

        for batch in loop:

            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, mask)

            loss = criterion(outputs, labels)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            loop.set_description(f"Epoch {epoch}")
            loop.set_postfix(loss=loss.item())

        acc, precision, recall, f1, val_loss = evaluate(model, val_loader)

        print("\nValidation Results")
        print("Accuracy:", acc)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1-score:", f1)
        print("Val Loss:", val_loss)

        # Save best model
        if f1 > best_f1:

            best_f1 = f1

            torch.save(
                model.state_dict(),
                "best_model.pt"
            )

            print("Best model saved")


    # =============================
    # Language Evaluation
    # =============================
    def evaluate_by_language(model, df):

        languages = df["lang"].unique()

        model.eval()

        for lang in languages:

            sub_df = df[df["lang"] == lang]

            dataset = SpamDataset(
                sub_df["text"].tolist(),
                sub_df["label"].tolist()
            )

            loader = DataLoader(dataset, batch_size=16)

            preds = []
            labels = []

            with torch.no_grad():

                for batch in loader:

                    input_ids = batch["input_ids"].to(device)
                    mask = batch["attention_mask"].to(device)
                    y = batch["label"].to(device)

                    output = model(input_ids, mask)

                    p = torch.argmax(output, dim=1)

                    preds.extend(p.cpu().numpy())
                    labels.extend(y.cpu().numpy())

            acc = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds)

            print("\nLanguage:", lang)
            print("Accuracy:", acc)
            print("F1-score:", f1)


    evaluate_by_language(model, df)

if __name__ == "__main__":
    main()