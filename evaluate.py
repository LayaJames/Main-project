@jax.jit
def predict(params, input_ids, attention_mask):
    logits = model.apply({"params": params}, input_ids, attention_mask)
    return jax.nn.sigmoid(logits).squeeze(-1)

predictions = predict(state.params, test_inputs, test_masks)
accuracy = np.mean((predictions > 0.5) == test_labels)
print(f"Test Accuracy: {accuracy}")
