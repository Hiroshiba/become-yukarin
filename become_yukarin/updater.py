import chainer
import numpy
from chainer import reporter

from .config import LossConfig
from .config import ModelConfig
from .model import Aligner
from .model import Discriminator
from .model import Predictor


class Updater(chainer.training.StandardUpdater):
    def __init__(
            self,
            loss_config: LossConfig,
            model_config: ModelConfig,
            predictor: Predictor,
            aligner: Aligner = None,
            discriminator: Discriminator = None,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.loss_config = loss_config
        self.model_config = model_config
        self.predictor = predictor
        self.aligner = aligner
        self.discriminator = discriminator

    def forward(self, input, target, mask):
        xp = self.predictor.xp

        input = chainer.as_variable(input)
        target = chainer.as_variable(target)
        mask = chainer.as_variable(mask)

        if self.aligner is not None:
            input = self.aligner(input)
        y = self.predictor(input)

        loss_l1 = chainer.functions.sum(chainer.functions.absolute_error(y, target) * mask)
        loss_l1 = loss_l1 / chainer.functions.sum(mask)
        reporter.report({'l1': loss_l1}, self.predictor)

        if self.discriminator is not None:
            pair_fake = chainer.functions.concat([y * mask, input])
            pair_true = chainer.functions.concat([target * mask, input])
            d_fake = self.discriminator(pair_fake)
            d_true = self.discriminator(pair_true)
            loss_dis_f = chainer.functions.mean_squared_error(d_fake, xp.zeros_like(d_fake.data, numpy.float32))
            loss_dis_t = chainer.functions.mean_squared_error(d_true, xp.ones_like(d_true.data, numpy.float32))
            loss_gen_f = chainer.functions.mean_squared_error(d_fake, xp.ones_like(d_fake.data, numpy.float32))
            reporter.report({'fake': loss_dis_f}, self.discriminator)
            reporter.report({'true': loss_dis_t}, self.discriminator)

        loss = {'predictor': loss_l1 * self.loss_config.l1}

        if self.aligner is not None:
            loss['aligner'] = loss_l1 * self.loss_config.l1
            reporter.report({'loss': loss['aligner']}, self.aligner)

        if self.discriminator is not None:
            loss['discriminator'] = \
                loss_dis_f * self.loss_config.discriminator_fake + \
                loss_dis_t * self.loss_config.discriminator_true
            reporter.report({'loss': loss['discriminator']}, self.discriminator)
            loss['predictor'] += loss_gen_f * self.loss_config.predictor_fake

        reporter.report({'loss': loss['predictor']}, self.predictor)
        return loss

    def update_core(self):
        batch = self.get_iterator('main').next()
        loss = self.forward(**self.converter(batch, self.device))

        for k, opt in self.get_all_optimizers().items():
            opt.update(loss.get, k)
