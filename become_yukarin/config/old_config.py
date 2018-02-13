from typing import List
from typing import NamedTuple
from typing import Optional


class CBHGDiscriminatorModelConfig(NamedTuple):
    in_channels: int
    hidden_channels_list: List[int]


class CBHGModelConfig(NamedTuple):
    in_channels: int
    conv_bank_out_channels: int
    conv_bank_k: int
    max_pooling_k: int
    conv_projections_hidden_channels: int
    highway_layers: int
    out_channels: int
    out_size: int
    aligner_out_time_length: int
    disable_last_rnn: bool
    enable_aligner: bool
    discriminator: Optional[CBHGDiscriminatorModelConfig]


class CBHGLossConfig(NamedTuple):
    l1: float
    predictor_fake: float
    discriminator_true: float
    discriminator_fake: float
    discriminator_grad: float
