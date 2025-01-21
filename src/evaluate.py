import jax
import jax.numpy as jnp
from flax.serialization import from_bytes
from transformers import BertTokenizer
from model import BERT_LSTM

# Initialize model
model = BERT_LSTM(lstm_hidden_dim=128, num_classes=2)

# Load model checkpoint
def load_model_checkpoint(model, model_path):
    rng = jax.random.PRNGKey(0)
    dummy_input_ids = jnp.ones((1, 6), dtype=jnp.int32)
    dummy_attention_mask = jnp.ones((1, 6), dtype=jnp.int32)

    with open(model_path, "rb") as f:
        trained_params = from_bytes(
            model.init({"params": rng}, dummy_input_ids, dummy_attention_mask, rng)["params"],
            f.read(),
        )

    state = model.init({"params": rng}, dummy_input_ids, dummy_attention_mask, rng)
    state = state.unfreeze()
    state["params"] = trained_params

    return state

# Evaluate model
def evaluate_model(state, test_data):
    input_ids = test_data["input_ids"]
    attention_mask = test_data["attention_mask"]
    labels = test_data["labels"]

    # Forward pass
    rng = jax.random.PRNGKey(42)
    logits = model.apply({"params": state["params"]}, input_ids, attention_mask, rng)
    predictions = jnp.argmax(logits, axis=-1)

    # Calculate accuracy
    accuracy = jnp.mean(predictions == labels)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Example usage
if __name__ == "__main__":
    model_path = "outputs/bert_lstm_model.pkl"
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Load and preprocess data
    input_ids = jnp.array([[101, 2023, 2003, 1037, 3231, 102]], dtype=jnp.int32)
    attention_masks = jnp.array([[1, 1, 1, 1, 1, 1]], dtype=jnp.int32)
    labels = jnp.array([1], dtype=jnp.int32)
    test_data = {"input_ids": input_ids, "attention_mask": attention_masks, "labels": labels}

    state = load_model_checkpoint(model, model_path)
    evaluate_model(state, test_data)
