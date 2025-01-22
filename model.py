from flax import linen as nn
import jax.numpy as jnp
import optax
from transformers import FlaxBertModel

class BertLSTMClassifier(nn.Module):
    hidden_size: int = 128
    num_classes: int = 1

    def setup(self):
        self.bert = FlaxBertModel.from_pretrained("bert-base-uncased", trainable=True)
        self.lstm = nn.scan(
            nn.LSTMCell,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=1,
            out_axes=1,
        )(hidden_size=self.hidden_size)
        self.classifier = nn.Dense(self.num_classes)

    def __call__(self, input_ids, attention_mask):
        # BERT embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_size)

        # LSTM layer
        lstm_out, _ = self.lstm(sequence_output)  # Shape: (batch_size, hidden_size)

        # Classification
        logits = self.classifier(lstm_out[:, -1])  # Use the last LSTM output
        return logits
