import pandas as pd
import jax.numpy as jnp
from transformers import BertTokenizer

def preprocess_data(file_path):
    data = pd.read_csv(file_path)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    input_ids = []
    attention_masks = []
    labels = data['label_column'].tolist()

    for text in data['text_column']:
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=64,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='np',  
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = jnp.array(input_ids)
    attention_masks = jnp.array(attention_masks)
    labels = jnp.array(labels)

    return input_ids, attention_masks, labels

if __name__ == "__main__":
    input_ids, attention_masks, labels = preprocess_data('data/data/payload_train.csv')
