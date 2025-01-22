from sklearn.model_selection import train_test_split

# Split the data
train_inputs, test_inputs, train_labels, test_labels = train_test_split(
    input_ids, labels, test_size=0.2, random_state=42
)
train_masks, test_masks = train_test_split(attention_mask, test_size=0.2, random_state=42)
