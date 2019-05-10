local embedding_dim = 768;
local hidden_dim = 32;

{
  "dataset_reader": {
    "type": "sst_tokens",
    "token_indexers": {
      "bert": {
        "type": "bert-pretrained",
        "pretrained_model": "bert-base-uncased"
      }
    }
  },
  "train_data_path": "data/stanfordSentimentTreebank/trees/train.txt",
  "validation_data_path": "data/stanfordSentimentTreebank/trees/dev.txt",

  "model": {
    "type": "lstm_classifier",

    "word_embeddings": {
      "allow_unmatched_keys": true,
      "bert": {
        "type": "bert-pretrained",
        "pretrained_model": "bert-base-uncased",
        "top_layer_only": true
      }
    },
    "encoder": {
      "type": "transformer",
      "input_dim": embedding_dim,
      "hidden_dim": hidden_dim,
      "projection_dim" : 64,
      "feedforward_hidden_dim" : 64,
      "num_layers" : 1,
      "num_attention_heads" : 4
    }
  },
  "iterator": {
    "type": "bucket",
    "batch_size": 256,
    "sorting_keys": [["tokens", "num_tokens"]]
  },
  "trainer": {
    "optimizer": {
      "type": "adam",
      "lr": 0.001
    },
    "num_epochs": 10,
    "patience": 5
  }
}
