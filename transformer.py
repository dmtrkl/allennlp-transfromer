import torch

from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.modules.seq2seq_encoders import StackedSelfAttentionEncoder

from allennlp.nn.util import get_final_encoder_states


@Seq2VecEncoder.register("transformer")
class TransformerSeq2VecEncoder(Seq2VecEncoder):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 projection_dim: int,
                 feedforward_hidden_dim: int,
                 num_layers: int,
                 num_attention_heads: int,
                 use_positional_encoding: bool = True,
                 dropout_prob: float = 0.1,
                 residual_dropout_prob: float = 0.2,
                 attention_dropout_prob: float = 0.1) -> None:

        super().__init__()

        self.seq2seq = StackedSelfAttentionEncoder(input_dim=input_dim,
                                                    hidden_dim=hidden_dim,
                                                    projection_dim=projection_dim,
                                                    feedforward_hidden_dim=feedforward_hidden_dim,
                                                    num_layers=num_layers,
                                                    num_attention_heads=num_attention_heads,
                                                    use_positional_encoding=use_positional_encoding,
                                                    dropout_prob=dropout_prob,
                                                    residual_dropout_prob=residual_dropout_prob,
                                                    attention_dropout_prob=attention_dropout_prob)

        self.hidden_dim = hidden_dim
        self.input_dim = input_dim


    def forward(self, inputs: torch.Tensor, mask: torch.Tensor):
        return get_final_encoder_states(self.seq2seq(inputs, None), mask)


    def get_input_dim(self):
        return self.input_dim

    def get_output_dim(self):
        return self.hidden_dim