from .config import LossConfig
from .model import Model

import chainer

from chainer import reporter


class Loss(chainer.link.Chain):
    def __init__(self, config: LossConfig, predictor: Model):
        super().__init__()
        self.config = config

        with self.init_scope():
            self.predictor = predictor

    def __call__(self, input, target):
        h = input
        y = self.predictor(h)

        loss = chainer.functions.mean_absolute_error(y, target)
        reporter.report({'loss': loss}, self)

        return loss * self.config.l1
