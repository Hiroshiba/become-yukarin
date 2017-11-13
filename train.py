import argparse
from pathlib import Path

from chainer.iterators import MultiprocessIterator
from chainer import optimizers
from chainer import training
from chainer.training import extensions
from chainer.dataset import convert

from become_yukarin.config import create_from_json
from become_yukarin.dataset import create as create_dataset
from become_yukarin.model import create as create_model
from become_yukarin.loss import Loss

from functools import partial

parser = argparse.ArgumentParser()
parser.add_argument('config_json_path', type=Path)
arguments = parser.parse_args()

config = create_from_json(arguments.config_json_path)
config.train.output.mkdir(exist_ok=True)
config.save_as_json((config.train.output / 'config.json').absolute())

# model
predictor = create_model(config.model)
model = Loss(config.loss, predictor=predictor)

# dataset
dataset = create_dataset(config.dataset)
train_iter = MultiprocessIterator(dataset['train'], config.train.batchsize)
test_iter = MultiprocessIterator(dataset['test'], config.train.batchsize, repeat=False, shuffle=False)
train_eval_iter = MultiprocessIterator(dataset['train_eval'], config.train.batchsize, repeat=False, shuffle=False)

# optimizer
optimizer = optimizers.Adam()
optimizer.setup(model)

# trainer
trigger_log = (config.train.log_iteration, 'iteration')
trigger_snapshot = (config.train.snapshot_iteration, 'iteration')

converter = partial(convert.concat_examples, padding=0)
updater = training.StandardUpdater(train_iter, optimizer, device=config.train.gpu, converter=converter)
trainer = training.Trainer(updater, out=config.train.output)

ext = extensions.Evaluator(test_iter, model, converter, device=config.train.gpu)
trainer.extend(ext, name='test', trigger=trigger_log)
ext = extensions.Evaluator(train_eval_iter, model, converter, device=config.train.gpu)
trainer.extend(ext, name='train', trigger=trigger_log)

trainer.extend(extensions.dump_graph('main/loss', out_name='graph.dot'))

ext = extensions.snapshot_object(predictor, filename='predictor_{.updater.iteration}.npz')
trainer.extend(ext, trigger=trigger_snapshot)

trainer.extend(extensions.LogReport(trigger=trigger_log, log_name='log.txt'))

if extensions.PlotReport.available():
    trainer.extend(extensions.PlotReport(
        y_keys=['main/loss', 'test/main/loss', 'train/main/loss'],
        x_key='iteration',
        file_name='loss.png',
        trigger=trigger_log,
    ))

trainer.run()
