"""
This code is forked from https://github.com/fomorians/td-gammon, I made some modification so it allows the
creation of multiple instances of the network
"""

import os
import time
import numpy as np

from utils import *


class SubNet:
    def set_strategy_name(self, name):
        self.STRATEGY = name

    def set_paths(self, model_path, summary_path, checkpoint_path):
        self.model_path = model_path + self.STRATEGY + '/'
        self.summary_path = summary_path + self.STRATEGY + '/'
        self.checkpoint_path = checkpoint_path + self.STRATEGY + '/'

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)

        if not os.path.exists(self.summary_path):
            os.makedirs(self.summary_path)

    def start_session(self, restore=False):
        graph = tf.Graph()
        session = tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
        with session.as_default(), graph.as_default():
            self.sess = session
            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            self.set_decay()
            self.set_nn()
            # after training a model, we can restore checkpoints here
            if restore:
                self.restore()

    def set_decay(self):
        # lambda decay
        self.lamda = tf.maximum(0.7, tf.train.exponential_decay(0.9, self.global_step,
                                                                30000, 0.96, staircase=True), name='lambda')

        # learning rate decay
        self.alpha = tf.maximum(0.01, tf.train.exponential_decay(0.1, self.global_step,
                                                                 40000, 0.96, staircase=True), name='alpha')

        tf.summary.scalar('lambda', self.lamda)
        tf.summary.scalar('alpha', self.alpha)

    def set_nn(self):
        # describe network size
        layer_size_input = 296
        layer_size_hidden = 50
        layer_size_output = 1

        # placeholders for input and target output
        self.x = tf.placeholder('float', [1, layer_size_input], name='x')
        self.V_next = tf.placeholder('float', [1, layer_size_output], name='V_next')

        # build network arch. (just 2 layers with sigmoid activation)
        prev_y = dense_layer(self.x, [layer_size_input, layer_size_hidden], tf.sigmoid, name='layer1')
        self.V = dense_layer(prev_y, [layer_size_hidden, layer_size_output], tf.sigmoid, name='layer2')

        # watch the individual value predictions over time
        tf.summary.scalar('V_next', tf.reduce_sum(self.V_next))
        tf.summary.scalar('V', tf.reduce_sum(self.V))

        # delta = V_next - V
        delta_op = tf.reduce_sum(self.V_next - self.V, name='delta')

        # mean squared error of the difference between the next state and the current state
        loss_op = tf.reduce_mean(tf.square(self.V_next - self.V), name='loss')

        # check if the model predicts the correct state
        accuracy_op = tf.reduce_sum(tf.cast(tf.equal(tf.round(self.V_next), tf.round(self.V)), dtype='float'),
                                    name='accuracy')

        # track the number of steps and average loss for the current game
        with tf.variable_scope('game'):
            game_step = tf.Variable(tf.constant(0.0), name='game_step', trainable=False)
            game_step_op = game_step.assign_add(1.0)

            loss_sum = tf.Variable(tf.constant(0.0), name='loss_sum', trainable=False)
            delta_sum = tf.Variable(tf.constant(0.0), name='delta_sum', trainable=False)
            accuracy_sum = tf.Variable(tf.constant(0.0), name='accuracy_sum', trainable=False)

            loss_avg_ema = tf.train.ExponentialMovingAverage(decay=0.999)
            delta_avg_ema = tf.train.ExponentialMovingAverage(decay=0.999)
            accuracy_avg_ema = tf.train.ExponentialMovingAverage(decay=0.999)

            loss_sum_op = loss_sum.assign_add(loss_op)
            delta_sum_op = delta_sum.assign_add(delta_op)
            accuracy_sum_op = accuracy_sum.assign_add(accuracy_op)

            loss_avg_op = loss_sum / tf.maximum(game_step, 1.0)
            delta_avg_op = delta_sum / tf.maximum(game_step, 1.0)
            accuracy_avg_op = accuracy_sum / tf.maximum(game_step, 1.0)

            loss_avg_ema_op = loss_avg_ema.apply([loss_avg_op])
            delta_avg_ema_op = delta_avg_ema.apply([delta_avg_op])
            accuracy_avg_ema_op = accuracy_avg_ema.apply([accuracy_avg_op])

            tf.summary.scalar('game/loss_avg', loss_avg_op)
            tf.summary.scalar('game/delta_avg', delta_avg_op)
            tf.summary.scalar('game/accuracy_avg', accuracy_avg_op)
            tf.summary.scalar('game/loss_avg_ema', loss_avg_ema.average(loss_avg_op))
            tf.summary.scalar('game/delta_avg_ema', delta_avg_ema.average(delta_avg_op))
            tf.summary.scalar('game/accuracy_avg_ema', accuracy_avg_ema.average(accuracy_avg_op))

            # reset per-game monitoring variables
            game_step_reset_op = game_step.assign(0.0)
            loss_sum_reset_op = loss_sum.assign(0.0)
            self.reset_op = tf.group(*[loss_sum_reset_op, game_step_reset_op])

        # increment global step: we keep this as a variable so it's saved with checkpoints
        global_step_op = self.global_step.assign_add(1)

        # get gradients of output V wrt trainable variables (weights and biases)
        tvars = tf.trainable_variables()
        grads = tf.gradients(self.V, tvars)

        # watch the weight and gradient distributions
        for grad, var in zip(grads, tvars):
            tf.summary.histogram(var.name, var)
            tf.summary.histogram(var.name + '/gradients/grad', grad)

        # for each variable, define operations to update the var with delta,
        # taking into account the gradient as part of the eligibility trace
        apply_gradients = []
        with tf.variable_scope('apply_gradients'):
            for grad, var in zip(grads, tvars):
                with tf.variable_scope('trace'):
                    # e-> = lambda * e-> + <grad of output w.r.t weights>
                    trace = tf.Variable(tf.zeros(grad.get_shape()), trainable=False, name='trace')
                    trace_op = trace.assign((self.lamda * trace) + grad)
                    tf.summary.histogram(var.name + '/traces', trace)

                # grad with trace = alpha * delta * e
                grad_trace = self.alpha * delta_op * trace_op
                tf.summary.histogram(var.name + '/gradients/trace', grad_trace)

                grad_apply = var.assign_add(grad_trace)
                apply_gradients.append(grad_apply)

        # as part of training we want to update our step and other monitoring variables
        with tf.control_dependencies([
            global_step_op,
            game_step_op,
            loss_sum_op,
            delta_sum_op,
            accuracy_sum_op,
            loss_avg_ema_op,
            delta_avg_ema_op,
            accuracy_avg_ema_op
        ]):
            # define single operation to apply all gradient updates
            self.train_op = tf.group(*apply_gradients, name='train')

        # merge summaries for TensorBoard
        self.summaries_op = tf.summary.merge_all()

        # create a saver for periodic checkpoints
        self.saver = tf.train.Saver(max_to_keep=1)

        # run variable initializers
        self.sess.run(tf.global_variables_initializer())

        self.summary_writer = tf.summary.FileWriter(
            '{0}{1}'.format(self.summary_path, int(time.time()), self.sess.graph_def))

    def restore(self):
        latest_checkpoint_path = tf.train.latest_checkpoint(self.checkpoint_path)
        if latest_checkpoint_path:
            print('Restoring checkpoint: {0}'.format(latest_checkpoint_path))
            self.saver.restore(self.sess, latest_checkpoint_path)

    def get_output(self, x):
        return self.sess.run(self.V, feed_dict={self.x: x})

    def run_output(self, x, V_next):
        self.sess.run(self.train_op, feed_dict={self.x: x, self.V_next: V_next})

    def create_model(self):
        tf.train.write_graph(self.sess.graph_def, self.model_path, self.STRATEGY + '_net.pb', as_text=False)

    def update_model(self, x, winner, episode, episodes, players, game_step):
        _, global_step, summaries, _ = self.sess.run([
            self.train_op,
            self.global_step,
            self.summaries_op,
            self.reset_op
        ], feed_dict={self.x: x, self.V_next: np.array([[winner]], dtype='float')})
        self.summary_writer.add_summary(summaries, global_step=global_step)

        print("Game %d/%d (Winner: %s) in %d turns" % (episode, episodes, players[winner].player, game_step))
        self.saver.save(self.sess, self.checkpoint_path + 'checkpoint', global_step=global_step)

    def training_end(self):
        self.summary_writer.close()
