# Fine-tune DistilBERT on CPU for the Transformer tagger

The Transformer tagger is DistilBERT (`distilbert-base-uncased`) fine-tuned with a multi-label head (sigmoid outputs, binary cross-entropy loss). It is trained on CPU, locally or on a GitHub-hosted runner, rather than BERT-base on a GPU. The only GPU available (GeForce MX130, 2 GB) is too small and too old for current PyTorch builds. Titles are short (around 32 tokens) and there are fewer than 5,000 Questions, so DistilBERT's 40% size reduction makes CPU training practical while keeping most of BERT's accuracy.

## Considered Options

- **BERT-base on Google Colab GPU**: rejected, because training would depend on an interactive notebook session outside the repository, which breaks reproducible training in CI.
- **Lighter non-transformer network (embeddings + LSTM/CNN)**: rejected, because the assignment explicitly suggests fine-tuning BERT, and a pretrained transformer is expected to generalise better on so little data.
