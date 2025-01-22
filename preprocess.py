from transformers import BertTokenizer

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenize the queries
def preprocess_texts(texts, max_length=128):
    return tokenizer(
        texts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="np"
    )

# Tokenize the dataset
tokenized_data = preprocess_texts(queries)
input_ids = tokenized_data["input_ids"]
attention_mask = tokenized_data["attention_mask"]
