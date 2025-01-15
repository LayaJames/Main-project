from transformers import BertModel
import torch.nn as nn

class BERT_LSTM(nn.Module):
    def __init__(self):
        super(BERT_LSTM, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.lstm = nn.LSTM(768, 128, batch_first=True)
        self.fc = nn.Linear(128, 2)  

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        lstm_output, (h_n, c_n) = self.lstm(sequence_output)
        logits = self.fc(lstm_output[:, -1, :])  
        return logits
