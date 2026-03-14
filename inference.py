import torch
from transformers import AutoTokenizer
from model import SpamModel

import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Load model
# =========================
model = SpamModel().to(device)

model.load_state_dict(
    torch.load(
        r"C:\Users\deamk\Downloads\RM\Model\best_model2.pt",
        map_location=device
    )
)

model.eval()
# =========================
# Tokenizer
# =========================
tokenizer = AutoTokenizer.from_pretrained(
    "xlm-roberta-base"
)


# =========================
# Test messages
# =========================
test_messages = [
    ("ENG", "Congratulations! you won free money click now"),
    ("TH", "ยินดีด้วย คุณได้รับเงินฟรี คลิกที่นี่"),
    ("ZH", "恭喜你中奖了 点击这里领取奖金")
]


# =========================
# Prediction function
# =========================
def predict(text):

    encoding = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )

    input_ids = encoding["input_ids"].to(device)
    mask = encoding["attention_mask"].to(device)

    with torch.no_grad():

        logits = model(input_ids, mask)

        probs = F.softmax(logits, dim=1)

        pred = torch.argmax(logits, dim=1).item()

        confidence = probs[0][pred].item()

    return pred, confidence


# =========================
# Run test
# =========================
for lang, text in test_messages:

    pred, conf = predict(text)

    print("\nLanguage:", lang)
    print("Text:", text)

    if pred == 1:
        print("Prediction: SPAM")
    else:
        print("Prediction: NORMAL")
    
    print("Confidence:", round(conf, 3))

