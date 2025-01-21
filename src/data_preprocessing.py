import pandas as pd
import numpy as np
import jax.numpy as jnp
from transformers import BertTokenizer
from sklearn.utils import resample
import jax.numpy as jnp

def preprocess_data(file_path, tokenizer, max_length=64):
    data = pd.read_csv(file_path)

    # Balance the dataset: 70% malicious, 30% non-malicious
    malicious = data[data['Label'] == 1]
    non_malicious = data[data['Label'] == 0]

    # Use the minimum of available rows and desired sample size
    malicious_sampled = resample(
        malicious,
        replace=False,
        n_samples=min(int(len(data) * 0.7), len(malicious)),
        random_state=42
    )
    non_malicious_sampled = resample(
        non_malicious,
        replace=False,
        n_samples=min(int(len(data) * 0.3), len(non_malicious)),
        random_state=42
    )

    balanced_data = pd.concat([malicious_sampled, non_malicious_sampled])

    input_ids, attention_masks, labels = [], [], balanced_data['Label'].tolist()
    for text in balanced_data['Query']:
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='np'
        )
        input_ids.append(encoded['input_ids'][0])
        attention_masks.append(encoded['attention_mask'][0])

    return jnp.array(input_ids), jnp.array(attention_masks), jnp.array(labels)

if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset_path = "Modified_SQL_Dataset.csv"
    input_ids, attention_masks, labels = preprocess_data(dataset_path, tokenizer)
    print("Preprocessing completed.")
