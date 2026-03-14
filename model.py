import torch
import torch.nn as nn
from transformers import AutoModel

class SpamModel(nn.Module):

    def __init__(self):#กำหนดโครง้สร้าง

        super().__init__()

        # XLM-R Encoder
        self.encoder = AutoModel.from_pretrained(
            "xlm-roberta-base"
        )

        hidden = 768

        # CNN branch จับ pattern spam
        self.cnn = nn.Conv1d(
            hidden,
            256,
            kernel_size=3,
            padding=1
        )

        # BiLSTM branch เรียนรู้ context
        self.lstm = nn.LSTM(
            hidden,
            256,
            batch_first=True,
            bidirectional=True
        )

        # Character CNN แบบพวก fr33
        self.char_embedding = nn.Embedding(128, 64)

        self.char_cnn = nn.Conv1d(
            64,
            128,
            kernel_size=3,
            padding=1
        )

        # classifier รวม feature
        self.fc = nn.Linear(
            256 + 512 + 128,
            128
        )

        self.dropout = nn.Dropout(0.3)

        self.out = nn.Linear(
            128,
            2
        )

    def forward(self, input_ids, attention_mask):

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        x = outputs.last_hidden_state

        # CNN branch #ใช้ในการจับ pattern ของคำที่มักจะเป็น spam เช่น "free", "win", "click"
        cnn_input = x.permute(0, 2, 1)
        cnn_out = self.cnn(cnn_input)
        cnn_out = torch.max(cnn_out, dim=2).values

        # BiLSTM branch ใช้ในการเรียนรู้ context ของข้อความ
        lstm_out, _ = self.lstm(x)
        lstm_out = torch.mean(lstm_out, dim=1)

        # Char branch (simple placeholder) จับ รูปแบบการสะกดคำ
        char = input_ids % 128
        char = self.char_embedding(char)
        char = char.permute(0, 2, 1)
        char = self.char_cnn(char)
        char = torch.max(char, dim=2).values

        # feature fusion
        features = torch.cat(
            [cnn_out, lstm_out, char],
            dim=1
        )

        x = self.fc(features)

        x = self.dropout(x)

        logits = self.out(x)

        return logits