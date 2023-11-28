# test-task1
This project performs training a named entity recognition (NER) model for the identification of mountain names inside the texts.
The dataset was generate with the help of ChatGPT. It consists of sentences and named entities.

At first, the dataset is prepared. Then tokenization is performed. After that, the tokenized text and labels are converted to tensor format. The data is splitted into training and validation sets, and these sets are wrapped in DataLoader. Finally, the fine-tuning setup is established, and the training process begins.
