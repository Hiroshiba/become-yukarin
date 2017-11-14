import chainer

from .config import ModelConfig


class Convolution1D(chainer.links.ConvolutionND):
    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0,
                 nobias=False, initialW=None, initial_bias=None,
                 cover_all=False):
        super(Convolution1D, self).__init__(
            ndim=1,
            in_channels=in_channels,
            out_channels=out_channels,
            ksize=ksize,
            stride=stride,
            pad=pad,
            nobias=nobias,
            initialW=initialW,
            initial_bias=initial_bias,
            cover_all=cover_all,
        )


class ConvHighway(chainer.link.Chain):
    def __init__(self, in_out_size, nobias=False, activate=chainer.functions.relu,
                 init_Wh=None, init_Wt=None, init_bh=None, init_bt=-1):
        super().__init__()
        self.activate = activate

        with self.init_scope():
            self.plain = Convolution1D(
                in_out_size, in_out_size, 1, nobias=nobias,
                initialW=init_Wh, initial_bias=init_bh)
            self.transform = Convolution1D(
                in_out_size, in_out_size, 1, nobias=nobias,
                initialW=init_Wt, initial_bias=init_bt)

    def __call__(self, x):
        out_plain = self.activate(self.plain(x))
        out_transform = chainer.functions.sigmoid(self.transform(x))
        y = out_plain * out_transform + x * (1 - out_transform)
        return y


class PreNet(chainer.link.Chain):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        with self.init_scope():
            self.conv1 = Convolution1D(in_channels, hidden_channels, 1)
            self.conv2 = Convolution1D(hidden_channels, out_channels, 1)

    def __call__(self, x):
        h = x
        h = chainer.functions.dropout((chainer.functions.relu(self.conv1(h)), 0.5))
        h = chainer.functions.dropout((chainer.functions.relu(self.conv2(h)), 0.5))
        return h


class Conv1DBank(chainer.link.Chain):
    def __init__(self, in_channels: int, out_channels: int, k: int):
        super().__init__()
        self.stacked_channels = out_channels * k
        self.pads = [
            chainer.functions.Pad(((0, 0), (0, 0), (i // 2, (i + 1) // 2)), mode='constant')
            for i in range(k)
        ]

        with self.init_scope():
            self.convs = chainer.link.ChainList(
                *(Convolution1D(in_channels, out_channels, i + 1, nobias=True) for i in range(k))
            )
            self.bn = chainer.links.BatchNormalization(out_channels * k)

    def __call__(self, x):
        h = x
        h = chainer.functions.concat([conv(pad(h)) for pad, conv in zip(self.pads, self.convs)])
        h = chainer.functions.relu(self.bn(h))
        return h


class Conv1DProjections(chainer.link.Chain):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()

        with self.init_scope():
            self.conv1 = Convolution1D(in_channels, hidden_channels, 3, pad=1, nobias=True)
            self.bn1 = chainer.links.BatchNormalization(hidden_channels)
            self.conv2 = Convolution1D(hidden_channels, out_channels, 3, pad=1, nobias=True)
            self.bn2 = chainer.links.BatchNormalization(out_channels)

    def __call__(self, x):
        h = x
        h = chainer.functions.relu(self.bn1(self.conv1(h)))
        h = chainer.functions.relu(self.bn2(self.conv2(h)))
        return h


class CBHG(chainer.link.Chain):
    def __init__(
            self,
            in_channels: int,
            conv_bank_out_channels: int,
            conv_bank_k: int,
            max_pooling_k: int,
            conv_projections_hidden_channels: int,
            highway_layers: int,
            out_channels: int,
    ):
        super().__init__()
        self.max_pooling_padding = chainer.functions.Pad(
            ((0, 0), (0, 0), ((max_pooling_k - 1) // 2, max_pooling_k // 2)),
            mode='constant',
        )
        self.max_pooling = chainer.functions.MaxPoolingND(1, max_pooling_k, 1, cover_all=False)
        self.out_size = out_channels

        with self.init_scope():
            self.conv_bank = Conv1DBank(
                in_channels=in_channels,
                out_channels=conv_bank_out_channels,
                k=conv_bank_k,
            )
            self.conv_projectoins = Conv1DProjections(
                in_channels=self.conv_bank.stacked_channels,
                hidden_channels=conv_projections_hidden_channels,
                out_channels=out_channels,
            )
            self.highways = chainer.link.ChainList(
                *([ConvHighway(out_channels) for _ in range(highway_layers)])
            )
            self.gru = chainer.links.NStepBiGRU(
                n_layers=1,
                in_size=out_channels,
                out_size=out_channels,
                dropout=0.0,
            )

    def __call__(self, x):
        h = x
        h = self.conv_bank(h)
        h = self.max_pooling(self.max_pooling_padding(h))
        h = self.conv_projectoins(h)
        h = h + x
        for highway in self.highways:
            h = highway(h)

        # h = chainer.functions.separate(chainer.functions.transpose(h, axes=(0, 2, 1)))
        # _, h = self.gru(None, h)
        # h = chainer.functions.transpose(chainer.functions.stack(h), axes=(0, 2, 1))
        return h


class Model(chainer.link.Chain):
    def __init__(self, network, out_size: int):
        super().__init__()
        with self.init_scope():
            self.network = network
            self.last = Convolution1D(network.out_size, out_size, 1)

    def __call__(self, x):
        h = x
        h = self.network(h)
        h = self.last(h)
        return h


def create(config: ModelConfig):
    network = CBHG(
        in_channels=config.in_channels,
        conv_bank_out_channels=config.conv_bank_out_channels,
        conv_bank_k=config.conv_bank_k,
        max_pooling_k=config.max_pooling_k,
        conv_projections_hidden_channels=config.conv_projections_hidden_channels,
        highway_layers=config.highway_layers,
        out_channels=config.out_channels,
    )
    model = Model(
        network=network,
        out_size=config.out_size,
    )
    return model
