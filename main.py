"""
This code is forked from https://github.com/fomorians/td-gammon
"""

import os
import tensorflow as tf

from modnet import Modnet

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_boolean('test', False, 'If true, test against a random strategy.')
flags.DEFINE_boolean('play', False, 'If true, play against a trained TD-Gammon strategy.')
flags.DEFINE_boolean('restore', False, 'If true, restore a checkpoint before training.')

model_path = os.environ.get('MODEL_PATH', 'models/')
summary_path = os.environ.get('SUMMARY_PATH', 'summaries/')
checkpoint_path = os.environ.get('CHECKPOINT_PATH', 'checkpoints/')

if not os.path.exists(model_path):
    os.makedirs(model_path)

if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

if not os.path.exists(summary_path):
    os.makedirs(summary_path)

if __name__ == '__main__':
    model = Modnet(model_path, summary_path, checkpoint_path, restore=FLAGS.restore)
    if FLAGS.test:
        model.test(episodes=1000)
    elif FLAGS.play:
        model.play()
    else:
        model.train()
