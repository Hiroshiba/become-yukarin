import chainer
import chainer.functions as F
from become_yukarin.config.sr_config import SRLossConfig

from become_yukarin.model.sr_model import SRDiscriminator
from become_yukarin.model.sr_model import SRPredictor


class SRUpdater(chainer.training.StandardUpdater):
    def __init__(
            self,
            loss_config: SRLossConfig,
            predictor: SRPredictor,
            discriminator: SRDiscriminator,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.loss_config = loss_config
        self.predictor = predictor
        self.discriminator = discriminator

    def _loss_predictor(self, predictor, output, target, d_fake):
        b, _, w, h = d_fake.data.shape

        loss_mse = (F.mean_absolute_error(output, target))
        chainer.report({'mse': loss_mse}, predictor)

        loss_adv = F.sum(F.softplus(-d_fake)) / (b * w * h)
        chainer.report({'adversarial': loss_adv}, predictor)

        loss = self.loss_config.mse * loss_mse + self.loss_config.adversarial * loss_adv
        chainer.report({'loss': loss}, predictor)
        return loss

    def _loss_discriminator(self, discriminator, y_in, y_out):
        b, _, w, h = y_in.data.shape

        loss_real = F.sum(F.softplus(-y_in)) / (b * w * h)
        chainer.report({'real': loss_real}, discriminator)

        loss_fake = F.sum(F.softplus(y_out)) / (b * w * h)
        chainer.report({'fake': loss_fake}, discriminator)

        loss = loss_real + loss_fake
        chainer.report({'loss': loss}, discriminator)
        return loss

    def forward(self, input, target):
        output = self.predictor(input)
        d_fake = self.discriminator(input, output)
        d_real = self.discriminator(input, target)

        loss = {
            'predictor': self._loss_predictor(self.predictor, output, target, d_fake),
            'discriminator': self._loss_discriminator(self.discriminator, d_real, d_fake),
        }
        return loss

    def update_core(self):
        opt_predictor = self.get_optimizer('predictor')
        opt_discriminator = self.get_optimizer('discriminator')

        batch = self.get_iterator('main').next()
        batch = self.converter(batch, self.device)
        loss = self.forward(**batch)

        opt_predictor.update(loss.get, 'predictor')
        opt_discriminator.update(loss.get, 'discriminator')
