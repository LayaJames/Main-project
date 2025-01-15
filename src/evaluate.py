import torch
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler
from model import BERT_LSTM
from data_preprocessing import preprocess_data

# Load and preprocess data
input_ids, attention_masks, labels = preprocess_data('data/data/payload_test.csv')
dataset = TensorDataset(input_ids, attention_masks, labels)
dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=32)

# Load trained model
model = BERT_LSTM()
model.load_state_dict(torch.load('path_to_saved_model.pt'))

# Evaluation loop
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in dataloader:
        batch_input_ids, batch_attention_masks, batch_labels = batch
        logits = model(batch_input_ids, batch_attention_masks)
        _, predicted = torch.max(logits, 1)
        total += batch_labels.size(0)
        correct += (predicted == batch_labels).sum().item()

print(f'Accuracy: {100 * correct / total}%')
