import chainer
import chainer.functions as F
import chainer.links as L

from become_yukarin.config.sr_config import SRModelConfig


class CBR(chainer.Chain):
    def __init__(self, ch0, ch1, bn=True, sample='down', activation=F.relu, dropout=False) -> None:
        super().__init__()
        self.bn = bn
        self.activation = activation
        self.dropout = dropout

        w = chainer.initializers.Normal(0.02)
        with self.init_scope():
            if sample == 'down':
                self.c = L.Convolution2D(ch0, ch1, 4, 2, 1, initialW=w)
            elif sample == 'up':
                self.c = L.Deconvolution2D(ch0, ch1, 4, 2, 1, initialW=w)
            else:
                self.c = L.Convolution2D(ch0, ch1, 1, 1, 0, initialW=w)
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


class SREncoder(chainer.Chain):
    def __init__(self, in_ch, base=64, extensive_layers=8) -> None:
        super().__init__()
        w = chainer.initializers.Normal(0.02)
        with self.init_scope():
            if extensive_layers > 0:
                self.c0 = L.Convolution2D(in_ch, base * 1, 3, 1, 1, initialW=w)
            else:
                self.c0 = L.Convolution2D(in_ch, base * 1, 1, 1, 0, initialW=w)

            _choose = lambda i: 'down' if i < extensive_layers else 'same'
            self.c1 = CBR(base * 1, base * 2, bn=True, sample=_choose(1), activation=F.leaky_relu, dropout=False)
            self.c2 = CBR(base * 2, base * 4, bn=True, sample=_choose(2), activation=F.leaky_relu, dropout=False)
            self.c3 = CBR(base * 4, base * 8, bn=True, sample=_choose(3), activation=F.leaky_relu, dropout=False)
            self.c4 = CBR(base * 8, base * 8, bn=True, sample=_choose(4), activation=F.leaky_relu, dropout=False)
            self.c5 = CBR(base * 8, base * 8, bn=True, sample=_choose(5), activation=F.leaky_relu, dropout=False)
            self.c6 = CBR(base * 8, base * 8, bn=True, sample=_choose(6), activation=F.leaky_relu, dropout=False)
            self.c7 = CBR(base * 8, base * 8, bn=True, sample=_choose(7), activation=F.leaky_relu, dropout=False)

    def __call__(self, x):
        hs = [F.leaky_relu(self.c0(x))]
        for i in range(1, 8):
            hs.append(self['c%d' % i](hs[i - 1]))
        return hs


class SRDecoder(chainer.Chain):
    def __init__(self, out_ch, base=64, extensive_layers=8) -> None:
        super().__init__()
        w = chainer.initializers.Normal(0.02)
        with self.init_scope():
            _choose = lambda i: 'up' if i >= 8 - extensive_layers else 'same'
            self.c0 = CBR(base * 8, base * 8, bn=True, sample=_choose(0), activation=F.relu, dropout=True)
            self.c1 = CBR(base * 16, base * 8, bn=True, sample=_choose(1), activation=F.relu, dropout=True)
            self.c2 = CBR(base * 16, base * 8, bn=True, sample=_choose(2), activation=F.relu, dropout=True)
            self.c3 = CBR(base * 16, base * 8, bn=True, sample=_choose(3), activation=F.relu, dropout=False)
            self.c4 = CBR(base * 16, base * 4, bn=True, sample=_choose(4), activation=F.relu, dropout=False)
            self.c5 = CBR(base * 8, base * 2, bn=True, sample=_choose(5), activation=F.relu, dropout=False)
            self.c6 = CBR(base * 4, base * 1, bn=True, sample=_choose(6), activation=F.relu, dropout=False)

            if extensive_layers > 0:
                self.c7 = L.Convolution2D(base * 2, out_ch, 3, 1, 1, initialW=w)
            else:
                self.c7 = L.Convolution2D(base * 2, out_ch, 1, 1, 0, initialW=w)

    def __call__(self, hs):
        h = self.c0(hs[-1])
        for i in range(1, 8):
            h = F.concat([h, hs[-i - 1]])
            if i < 7:
                h = self['c%d' % i](h)
            else:
                h = self.c7(h)
        return h


class SRPredictor(chainer.Chain):
    def __init__(self, in_ch, out_ch, base, extensive_layers) -> None:
        super().__init__()
        with self.init_scope():
            self.encoder = Encoder(in_ch, base=base, extensive_layers=extensive_layers)
            self.decoder = Decoder(out_ch, base=base, extensive_layers=extensive_layers)

    def __call__(self, x):
        return self.decoder(self.encoder(x))


class SRDiscriminator(chainer.Chain):
    def __init__(self, in_ch, out_ch, base=32, extensive_layers=5) -> None:
        super().__init__()
        w = chainer.initializers.Normal(0.02)
        with self.init_scope():
            _choose = lambda i: 'down' if i < extensive_layers else 'same'
            self.c0_0 = CBR(in_ch, base * 1, bn=False, sample=_choose(0), activation=F.leaky_relu, dropout=False)
            self.c0_1 = CBR(out_ch, base * 1, bn=False, sample=_choose(0), activation=F.leaky_relu, dropout=False)
            self.c1 = CBR(base * 2, base * 4, bn=True, sample=_choose(1), activation=F.leaky_relu, dropout=False)
            self.c2 = CBR(base * 4, base * 8, bn=True, sample=_choose(2), activation=F.leaky_relu, dropout=False)
            self.c3 = CBR(base * 8, base * 16, bn=True, sample=_choose(3), activation=F.leaky_relu, dropout=False)

            if extensive_layers > 4:
                self.c4 = L.Convolution2D(base * 16, 1, 3, 1, 1, initialW=w)
            else:
                self.c4 = L.Convolution2D(base * 16, 1, 1, 1, 0, initialW=w)

    def __call__(self, x_0, x_1):
        h = F.concat([self.c0_0(x_0), self.c0_1(x_1)])
        h = self.c1(h)
        h = self.c2(h)
        h = self.c3(h)
        h = self.c4(h)
        # h = F.average_pooling_2d(h, h.data.shape[2], 1, 0)
        return h


def create_predictor_sr(config: SRModelConfig):
    return SRPredictor(
        in_ch=1,
        out_ch=1,
        base=config.generator_base_channels,
        extensive_layers=config.generator_extensive_layers,
    )


def create_discriminator_sr(config: SRModelConfig):
    return SRDiscriminator(
        in_ch=1,
        out_ch=1,
        base=config.discriminator_base_channels,
        extensive_layers=config.discriminator_extensive_layers,
    )


def create_sr(config: SRModelConfig):
    predictor = create_predictor_sr(config)
    discriminator = create_discriminator_sr(config)
    return predictor, discriminator
