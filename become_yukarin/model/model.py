import chainer
import chainer.functions as F
import chainer.links as L

from become_yukarin.config.config import ModelConfig


class Convolution1D(chainer.links.ConvolutionND):
    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0,
                 nobias=False, initialW=None, initial_bias=None,
                 cover_all=False) -> None:
        super().__init__(
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


class Deconvolution1D(chainer.links.DeconvolutionND):
    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0,
                 nobias=False, outsize=None,
                 initialW=None, initial_bias=None) -> None:
        super().__init__(
            ndim=1,
            in_channels=in_channels,
            out_channels=out_channels,
            ksize=ksize,
            stride=stride,
            pad=pad,
            nobias=nobias,
            outsize=outsize,
            initialW=initialW,
            initial_bias=initial_bias,
        )


class CBR(chainer.Chain):
    def __init__(self, ch0, ch1, bn=True, sample='down', activation=F.relu, dropout=False) -> None:
        super().__init__()
        self.bn = bn
        self.activation = activation
        self.dropout = dropout

        w = chainer.initializers.Normal(0.02)
        with self.init_scope():
            if sample == 'down':
                self.c = Convolution1D(ch0, ch1, 4, 2, 1, initialW=w)
            else:
                self.c = Deconvolution1D(ch0, ch1, 4, 2, 1, initialW=w)
            if bn:
                self.batchnorm = L.BatchNormalization(ch1)

    def __call__(self, x):
        h = self.c(x)
        if self.bn:
            h = self.batchnorm(h)
        if self.dropout:
            h = F.dropout(h)
        if self.activation is not None:
            h = self.activation(h)
        return h


class Encoder(chainer.Chain):
    def __init__(self, in_ch) -> None:
        super().__init__()
        w = chainer.initializers.Normal(0.02)
        with self.init_scope():
            self.c0 = Convolution1D(in_ch, 64, 3, 1, 1, initialW=w)
            self.c1 = CBR(64, 128, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
            self.c2 = CBR(128, 256, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
            self.c3 = CBR(256, 512, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
            self.c4 = CBR(512, 512, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
            self.c5 = CBR(512, 512, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
            self.c6 = CBR(512, 512, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
            self.c7 = CBR(512, 512, bn=True, sample='down', activation=F.leaky_relu, dropout=False)

    def __call__(self, x):
        hs = [F.leaky_relu(self.c0(x))]
        for i in range(1, 8):
            hs.append(self['c%d' % i](hs[i - 1]))
        return hs


class Decoder(chainer.Chain):
    def __init__(self, out_ch) -> None:
        super().__init__()
        w = chainer.initializers.Normal(0.02)
        with self.init_scope():
            self.c0 = CBR(512, 512, bn=True, sample='up', activation=F.relu, dropout=True)
            self.c1 = CBR(1024, 512, bn=True, sample='up', activation=F.relu, dropout=True)
            self.c2 = CBR(1024, 512, bn=True, sample='up', activation=F.relu, dropout=True)
            self.c3 = CBR(1024, 512, bn=True, sample='up', activation=F.relu, dropout=False)
            self.c4 = CBR(1024, 256, bn=True, sample='up', activation=F.relu, dropout=False)
            self.c5 = CBR(512, 128, bn=True, sample='up', activation=F.relu, dropout=False)
            self.c6 = CBR(256, 64, bn=True, sample='up', activation=F.relu, dropout=False)
            self.c7 = Convolution1D(128, out_ch, 3, 1, 1, initialW=w)

    def __call__(self, hs):
        h = self.c0(hs[-1])
        for i in range(1, 8):
            h = F.concat([h, hs[-i - 1]])
            if i < 7:
                h = self['c%d' % i](h)
            else:
                h = self.c7(h)
        return h


class Predictor(chainer.Chain):
    def __init__(self, in_ch, out_ch) -> None:
        super().__init__()
        with self.init_scope():
            self.encoder = Encoder(in_ch)
            self.decoder = Decoder(out_ch)

    def __call__(self, x):
        return self.decoder(self.encoder(x))


class Discriminator(chainer.Chain):
    def __init__(self, in_ch, out_ch) -> None:
        super().__init__()
        w = chainer.initializers.Normal(0.02)
        with self.init_scope():
            self.c0_0 = CBR(in_ch, 32, bn=False, sample='down', activation=F.leaky_relu, dropout=False)
            self.c0_1 = CBR(out_ch, 32, bn=False, sample='down', activation=F.leaky_relu, dropout=False)
            self.c1 = CBR(64, 128, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
            self.c2 = CBR(128, 256, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
            self.c3 = CBR(256, 512, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
            self.c4 = Convolution1D(512, 1, 3, 1, 1, initialW=w)

    def __call__(self, x_0, x_1):
        h = F.concat([self.c0_0(x_0), self.c0_1(x_1)])
        h = self.c1(h)
        h = self.c2(h)
        h = self.c3(h)
        h = self.c4(h)
        # h = F.average_pooling_2d(h, h.data.shape[2], 1, 0)
        return h


def create_predictor(config: ModelConfig):
    return Predictor(in_ch=config.in_channels, out_ch=config.out_channels)


def create(config: ModelConfig):
    predictor = create_predictor(config)
    discriminator = Discriminator(in_ch=config.in_channels, out_ch=config.out_channels)
    return predictor, discriminator
