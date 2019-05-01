"""
This code is forked from https://github.com/fomorians/td-gammon
"""

import tensorflow as tf

from tester import *

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_boolean('test', False, 'If true, run a test')
flags.DEFINE_boolean('all', False, 'If true, test against a random strategy.')
flags.DEFINE_boolean('best', False, 'If true, test against a random strategy.')
flags.DEFINE_boolean('play', False, 'If true, play against a trained TD-Gammon strategy.')
flags.DEFINE_boolean('restore', True, 'If true, restore a checkpoint before training.')
flags.DEFINE_boolean('mono', False, 'If true, use monolithic NN.')
flags.DEFINE_boolean('hybrid', False, 'If true, use Modular Network with Hybrid strategy.')
flags.DEFINE_boolean('draw', False, 'If true, use Modular Network with Hybrid strategy.')

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
    model_mod_hybrid = ModnetHybrid(model_path, summary_path, checkpoint_path, restore=FLAGS.restore)
    model_mono = MonoNN(model_path, summary_path, checkpoint_path, restore=FLAGS.restore)
    create_plots()
    # if FLAGS.test and FLAGS.all:
    #     if FLAGS.best:
    #         if FLAGS.mono:
    #             test_all_best(model_mono, draw=FLAGS.draw)
    #         elif FLAGS.hybrid:
    #             test_all_best(model_mod_hybrid, draw=FLAGS.draw)
    #         else:
    #             test_all_best(model_mod, draw=FLAGS.draw)
    #     else:
    #         if FLAGS.mono:
    #             test_all_random(model_mono, draw=FLAGS.draw)
    #         elif FLAGS.hybrid:
    #             test_all_random(model_mod_hybrid, draw=FLAGS.draw)
    #         else:
    #             test_all_random(model_mod, draw=FLAGS.draw)
    # elif FLAGS.test:
    #     if FLAGS.mono:
    #         test_random(model_mono, episodes=1000, draw=FLAGS.draw)
    #     elif FLAGS.hybrid:
    #         test_random(model_mod_hybrid, episodes=1000, draw=FLAGS.draw)
    #     else:
    #         test_random(model_mod, episodes=1000, draw=FLAGS.draw)
    # elif FLAGS.play:
    #     if FLAGS.mono:
    #         model_mono.play()
    #     elif FLAGS.hybrid:
    #         model_mod_hybrid.play()
    #     else:
    #         model_mod.play()
    # else:
    #     if FLAGS.mono:
    #         model_mono.train(episodes=1000000)
    #     elif FLAGS.hybrid:
    #         model_mod_hybrid.train(episodes=1000000)
    #     else:
    #         model_mod.train(episodes=1000000)
