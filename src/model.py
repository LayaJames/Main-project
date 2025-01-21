import jax
import jax.numpy as jnp
import flax.linen as nn
from transformers import FlaxBertModel


class BERT_LSTM(nn.Module):
    lstm_hidden_dim: int  # Hidden state dimension for LSTM
    num_classes: int       # Number of output classes

    def setup(self):
        # Load pre-trained BERT model without pooling layer
        self.bert = FlaxBertModel.from_pretrained("bert-base-uncased", add_pooling_layer=False)
        # Define LSTM layer
        self.lstm = nn.LSTMCell(features=self.lstm_hidden_dim)
        # Fully connected layer for classification
        self.fc = nn.Dense(self.num_classes)

    def __call__(self, input_ids, attention_mask, *, rng=None):  # 'rng' is optional for initialization
        # Extract embeddings from BERT
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_output.last_hidden_state  # Shape: (batch_size, seq_length, hidden_size)

        if rng is None:
            raise ValueError("RNG must be provided for the forward pass.")

        # Initialize the LSTM carry state
        batch_size, seq_length, hidden_size = sequence_output.shape
        lstm_state = self.lstm.initialize_carry(rng, (batch_size,), self.lstm_hidden_dim)

        # Process the sequence output through the LSTM for each token in the sequence
        for t in range(seq_length):
            lstm_state = self.lstm(sequence_output[:, t, :], lstm_state)

        # Extract the final hidden state
        final_hidden = lstm_state[0]  # lstm_state[0] is the hidden state (hx)
        logits = self.fc(final_hidden)  # Shape: (batch_size, num_classes)

        return logits


# Example usage
def main():
    # Random number generator key for initialization
    rng = jax.random.PRNGKey(0)
    model = BERT_LSTM(lstm_hidden_dim=128, num_classes=2)

    # Example tokenized and preprocessed data (input_ids and attention_mask)
    dummy_input_ids = jnp.array([
        [101, 2023, 2003, 1037, 3231, 102],
        [101, 2023, 2003, 1037, 2332, 102]
    ], dtype=jnp.int32)
    dummy_attention_mask = jnp.array([
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1]
    ], dtype=jnp.int32)

    # Split the random key
    rng, init_rng = jax.random.split(rng)

    # Initialize model parameters (Flax will handle `rng` internally for `init`)
    params = model.init(init_rng, dummy_input_ids, dummy_attention_mask)

    # Apply the model (provide `rng` explicitly)
    logits = model.apply({'params': params}, dummy_input_ids, dummy_attention_mask, rng=rng)
    print("Logits:", logits)


# if __name__ == "__main__":
    main()
