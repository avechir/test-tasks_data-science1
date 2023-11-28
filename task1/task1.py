import torch
from transformers import BertTokenizer, BertForTokenClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split

dataset = [
    ("Mount Everest is the highest peak in the world.", {"entities": [(0, 13, "Mountain")]}),
    ("Kilimanjaro is the tallest mountain in Africa.", {"entities": [(0, 11, "Mountain")]}),
    ("Denali, also known as Mount McKinley, is in North America.", {"entities": [(0, 6, "Mountain"), (21, 35, "Mountain")]}),
    ("The Matterhorn is a famous mountain in the Alps.", {"entities": [(4, 14, "Mountain")]}),
    ("Mount Fuji is an iconic volcano in Japan.", {"entities": [(0, 10, "Mountain")]}),
    ("The Rocky Mountains stretch across North America.", {"entities": [(4, 18, "Mountain")]}),
    ("The Himalayas are a vast mountain range in Asia.", {"entities": [(4, 12, "Mountain")]}),
    ("Mount Elbrus is the highest mountain in Europe.", {"entities": [(0, 11, "Mountain")]})
]

# Preparing dataset
sentences = []
labels = []
for sentence, label in dataset:
    sentences.append(sentence)
    labels.append(label['entities'])

# Loading the pre-trained tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenization
def tokenize_and_preserve_labels(sentence, text_labels):
    tokenized_sentence = []
    labels = []

    for word, label in zip(sentence, text_labels):
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)
        tokenized_sentence.extend(tokenized_word)
        labels.extend([label] * n_subwords)

    return tokenized_sentence, labels

tokenized_texts_and_labels = [tokenize_and_preserve_labels(sent, labs) for sent, labs in zip(sentences, labels)]
tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]

# Convertion of labels to a binary format
model_labels = []
for entities in labels:
    model_label = [0] * len(tokenized_texts)  
    for start, end, _ in entities:
        for position in range(len(tokenized_texts)):
            if start <= position < end:
                model_label[position] = 1
    model_labels.append(model_label)
print(model_labels)

model_labels_flat = [label for sublist in model_labels for label in sublist]

# Creation of mapping from labels to indexes
tag2idx = {tag: idx for idx, tag in enumerate(set(model_labels_flat))}
tag2idx["PAD"] = len(tag2idx)
# Convertion to tensor format
input_ids = [tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts]
padded_input_ids = pad_sequence([torch.tensor(ids) for ids in input_ids], batch_first=True, padding_value=0)

# Creation of attention masks to ignore padded tokens
attention_masks = [[float(i != 0.0) for i in ii] for ii in padded_input_ids]

# splitting data into train and validation sets
train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(padded_input_ids, model_labels, random_state=42, test_size=0.1)
train_masks, validation_masks, _, _ = train_test_split(attention_masks, padded_input_ids, random_state=42, test_size=0.1)

# Convertion to PyTorch tensors
train_inputs = torch.tensor(train_inputs)
validation_inputs = torch.tensor(validation_inputs)
train_labels = torch.tensor(train_labels)
validation_labels = torch.tensor(validation_labels)
train_masks = torch.tensor(train_masks)
validation_masks = torch.tensor(validation_masks)
print(train_labels)
# Create TensorDatasets
train_dataset = TensorDataset(train_inputs, train_labels, train_masks)
validation_dataset = TensorDataset(validation_inputs, validation_labels, validation_masks)
# Create DataLoaders
batch_size = 2
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
# Model loading
model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=2)
# Fine-tuning setup
epoches = 5
optimizer = AdamW(model.parameters(), lr=3e-5, eps=1e-8)
total_steps = len(train_dataloader) * epoches
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
# Training
print(len(train_dataloader))
for _ in range(epoches):
    model.train()
    for batch in train_dataloader:
        b_input_ids, b_labels, b_masks = batch
        print(b_input_ids)
        print(b_labels)
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_masks, labels=b_labels)
        loss = outputs[0]
        loss.backward()
        optimizer.step()
        scheduler.step()
        model.zero_grad()

model.save_pretrained("./ner_model")