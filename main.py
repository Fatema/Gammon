"""
This code is forked from https://github.com/fomorians/td-gammon
"""

import tensorflow as tf

from tester import *

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_boolean('test', False, 'If true, test against a random strategy.')
flags.DEFINE_boolean('play', False, 'If true, play against a trained TD-Gammon strategy.')
flags.DEFINE_boolean('restore', False, 'If true, restore a checkpoint before training.')
flags.DEFINE_boolean('mono', False, 'If true, use monolithic NN.')

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
    model_mod = Modnet(model_path, summary_path, checkpoint_path, restore=FLAGS.restore)
    model_mono = MonoNN(model_path, summary_path, checkpoint_path, restore=FLAGS.restore)
    if FLAGS.test:
        if FLAGS.mono:
            test_self(model_mono, episodes=1000)
        else:
            test_random(model_mod, episodes=1000)
    elif FLAGS.play:
        if FLAGS.mono:
            model_mono.play()
        else:
            model_mod.play()
    else:
        if FLAGS.mono:
            model_mono.train(episodes=500000)
        else:
            model_mod.train()
