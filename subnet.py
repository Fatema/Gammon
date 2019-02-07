"""
This code is forked from https://github.com/fomorians/td-gammon, I made some modification so it allows the
creation of multiple instances of the network
"""

import os
import time
import numpy as np
from tensorflow.python.tools import inspect_checkpoint as chkp

from utils import *


class SubNet:
    def set_network_name(self, name):
        self.STRATEGY = name

    def set_paths(self, model_path, summary_path, checkpoint_path):
        self.model_path = model_path + self.STRATEGY + '/'
        self.summary_path = summary_path + self.STRATEGY + '/'
        self.checkpoint_path = checkpoint_path + self.STRATEGY + '/latest/'
        self.previous_checkpoint_path = checkpoint_path + self.STRATEGY + '/previous/'
        self.test_checkpoint_path = checkpoint_path + self.STRATEGY + '/test/'


        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)

        if not os.path.exists(self.previous_checkpoint_path):
            os.makedirs(self.previous_checkpoint_path)

        if not os.path.exists(self.test_checkpoint_path):
            os.makedirs(self.test_checkpoint_path)

        if not os.path.exists(self.summary_path):
            os.makedirs(self.summary_path)

    def start_session(self, restore=False):
        graph = tf.Graph()
        session = tf.Session(graph=graph)
        with session.as_default(), graph.as_default():
            self.sess = session
            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            self.set_decay()
            self.set_nn()
            # after training a model, we can restore checkpoints here
            if restore:
                self.restore()
            else:
                self.set_previous_checkpoint()

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
        # set input layer such that each field is 4 units
        layer_size_input = 200
        layer_size_hidden = 50
        # set output layer to 4 units
        # 2 represent the odds of white or black winning
        # 2 represent either getting a gammon
        layer_size_output = 4

        # placeholders for input and target output
        self.x = tf.placeholder('float', [1, layer_size_input], name='x')
        self.V_next = tf.placeholder('float', [1, layer_size_output], name='V_next')

        # build network arch. (just 2 layers with sigmoid activation)
        self.prev_y = dense_layer(self.x, [layer_size_input, layer_size_hidden], tf.sigmoid, name='layer1')
        self.V = dense_layer(self.prev_y, [layer_size_hidden, layer_size_output], tf.sigmoid, name='layer2')

        # watch the individual value predictions over time
        tf.summary.scalar('o_next', self.V_next[0][0])
        tf.summary.scalar('o_gammon_next', self.V_next[0][2])
        tf.summary.scalar('x_next', self.V_next[0][1])
        tf.summary.scalar('x_gammon_next', self.V_next[0][3])
        tf.summary.scalar('o', self.V[0][0])
        tf.summary.scalar('o_gammon', self.V[0][2])
        tf.summary.scalar('x', self.V[0][1])
        tf.summary.scalar('x_gammon', self.V[0][3])

        # delta = V_next - V
        delta_op = self.V_next - self.V

        # mean squared error of the difference between the next state and the current state
        loss_op = tf.reduce_mean(tf.square(delta_op), name='loss')

        # track the number of steps and average loss for the current game
        with tf.variable_scope('game'):
            game_step = tf.Variable(tf.constant(0.0), name='game_step', trainable=False)
            game_step_op = game_step.assign_add(1.0)

            loss_sum = tf.Variable(tf.constant(0.0), name='loss_sum', trainable=False)
            delta_sum = tf.Variable(tf.constant(0.0), name='delta_sum', trainable=False)

            loss_avg_ema = tf.train.ExponentialMovingAverage(decay=0.999)
            delta_avg_ema = tf.train.ExponentialMovingAverage(decay=0.999)

            loss_sum_op = loss_sum.assign_add(loss_op)
            delta_sum_op = delta_sum.assign_add(delta_op)

            loss_avg_op = loss_sum / tf.maximum(game_step, 1.0)
            delta_avg_op = delta_sum / tf.maximum(game_step, 1.0)

            loss_avg_ema_op = loss_avg_ema.apply([loss_avg_op])
            delta_avg_ema_op = delta_avg_ema.apply([delta_avg_op])

            tf.summary.scalar('game/loss_avg', loss_avg_op)
            tf.summary.scalar('game/delta_avg', delta_avg_op)
            tf.summary.scalar('game/loss_avg_ema', loss_avg_ema.average(loss_avg_op))
            tf.summary.scalar('game/delta_avg_ema', delta_avg_ema.average(delta_avg_op))

            # reset per-game monitoring variables
            game_step_reset_op = game_step.assign(0.0)
            loss_sum_reset_op = loss_sum.assign(0.0)
            self.reset_op = tf.group(*[loss_sum_reset_op, game_step_reset_op])

        # increment global step: we keep this as a variable so it's saved with checkpoints
        global_step_op = self.global_step.assign_add(1)

        # get gradients of output V wrt trainable variables (weights and biases)
        tvars = tf.trainable_variables()
        grads = self.manual_gradients(tvars)

        # watch the weight and gradient distributions
        for grad, var in zip(grads, tvars):
            print(grad)
            tf.summary.histogram(var.name, var)
            tf.summary.histogram(var.name + '/gradients/grad', grad)

        # for each variable, define operations to update the var with delta,
        # taking into account the gradient as part of the eligibility trace
        apply_gradients = []
        with tf.variable_scope('apply_gradients'):
            for grad, var in zip(grads, tvars):
                print(var)
                with tf.variable_scope('trace'):
                    # e-> = lambda * e-> + <grad of output w.r.t weights>
                    trace = tf.Variable(tf.zeros(grad.get_shape()), trainable=False, name='trace')
                    trace_op = trace.assign((self.lamda * trace) + grad)
                    tf.summary.histogram(var.name + '/traces', trace)

                # grad with trace = alpha * delta * e
                if 'layer1' in var.name:
                    grad_trace = self.alpha * tf.reduce_sum(tf.multiply(delta_op, trace_op), axis=len(trace_op.get_shape()) - 1)
                elif 'layer2/bias' in var.name:
                    grad_trace = self.alpha * tf.reduce_sum(delta_op * trace_op, axis=0)
                else:
                    grad_trace = self.alpha * delta_op * trace_op

                tf.summary.histogram(var.name + '/gradients/trace', grad_trace)

                print('grade trace', grad_trace.get_shape(), delta_op.get_shape(), trace_op.get_shape())

                grad_apply = var.assign_add(grad_trace)
                apply_gradients.append(grad_apply)

        # as part of training we want to update our step and other monitoring variables
        with tf.control_dependencies([
            global_step_op,
            game_step_op,
            loss_sum_op,
            delta_sum_op,
            loss_avg_ema_op,
            delta_avg_ema_op,
        ]):
            # define single operation to apply all gradient updates
            self.train_op = tf.group(*apply_gradients, name='train')

        # merge summaries for TensorBoard
        self.summaries_op = tf.summary.merge_all()

        # create a saver for periodic checkpoints
        self.saver = tf.train.Saver(max_to_keep=1)
        self.pre_saver = tf.train.Saver(max_to_keep=1)
        self.testing_saver = tf.train.Saver(max_to_keep=None)

        # run variable initializers
        self.sess.run(tf.global_variables_initializer())

        self.summary_writer = tf.summary.FileWriter(
            '{0}{1}'.format(self.summary_path, int(time.time()), self.sess.graph_def))

    def restore(self):
        latest_checkpoint_path = tf.train.latest_checkpoint(self.checkpoint_path)
        if latest_checkpoint_path:
            print('Restoring checkpoint: {0}'.format(latest_checkpoint_path))
            self.saver.restore(self.sess, latest_checkpoint_path)
            chkp.print_tensors_in_checkpoint_file(latest_checkpoint_path, tensor_name='', all_tensors=True)

    def restore_previous(self):
        print('restoring previous')
        latest_checkpoint_path = tf.train.latest_checkpoint(self.previous_checkpoint_path)
        if latest_checkpoint_path:
            print('Restoring previous checkpoint: {0}'.format(latest_checkpoint_path))
            self.pre_saver.restore(self.sess, latest_checkpoint_path)
            chkp.print_tensors_in_checkpoint_file(latest_checkpoint_path, tensor_name='', all_tensors=True)

    def set_previous_checkpoint(self):
        self.pre_saver.save(self.sess, self.previous_checkpoint_path + 'checkpoint', global_step=self.global_step)

    def set_test_checkpoint(self):
        self.pre_saver.save(self.sess, '{0}{1}/{2}'.format(self.test_checkpoint_path, int(time.time()), 'checkpoint'), global_step=self.global_step)

    def restore_test_checkpoint(self, timestamp):
        print('restoring previous')
        latest_checkpoint_path = tf.train.latest_checkpoint('{0}{1}/'.format(self.test_checkpoint_path, timestamp))
        print(latest_checkpoint_path)
        # todo figure out a strategy to run the tests for multiple checkpoints

    def print_checkpoints(self):
        latest_checkpoint_path = tf.train.latest_checkpoint(self.checkpoint_path)
        previous_checkpoint_path = tf.train.latest_checkpoint(self.previous_checkpoint_path)
        chkp.print_tensors_in_checkpoint_file(latest_checkpoint_path, tensor_name='', all_tensors=True)
        chkp.print_tensors_in_checkpoint_file(previous_checkpoint_path, tensor_name='', all_tensors=True)

    def get_output(self, x):
        return self.sess.run(self.V, feed_dict={self.x: x})

    def run_output(self, x, V_next):
        self.sess.run(self.train_op, feed_dict={self.x: x, self.V_next: V_next})

    def create_model(self):
        tf.train.write_graph(self.sess.graph_def, self.model_path, self.STRATEGY + '_net.pb', as_text=False)

    def update_model(self, x, out, episode, episodes, players, game_step):
        _, global_step, summaries, _ = self.sess.run([
            self.train_op,
            self.global_step,
            self.summaries_op,
            self.reset_op
        ], feed_dict={self.x: x, self.V_next: out})
        self.summary_writer.add_summary(summaries, global_step=global_step)

        p = 0 if out[0][0] else 1

        print("Game %d/%d (Winner: %s) in %d turns" % (episode, episodes, players[p].player, game_step))
        self.saver.save(self.sess, self.checkpoint_path + 'checkpoint', global_step=global_step)

    def training_end(self):
        self.summary_writer.close()

    def manual_gradients(self, tvars):
        _, _, w2, _ = tvars
        dw1_iho = self.x[0][:, np.newaxis, np.newaxis] * (self.prev_y * (1 - self.prev_y))[0][np.newaxis, :, np.newaxis] * w2[np.newaxis, :, :] * \
              (self.V * (1 - self.V))[0][np.newaxis, np.newaxis, :]
        db1_ho = (self.prev_y * (1 - self.prev_y))[0][:, np.newaxis] * w2[:, :] * \
              (self.V * (1 - self.V))[0][np.newaxis, :]
        dw2_ho = self.prev_y[0][:, np.newaxis] * (self.V * (1 - self.V))[0][np.newaxis, :]
        db2_o = self.V * (1 - self.V)
        return [dw1_iho, db1_ho, dw2_ho, db2_o]

    def tf_gradients(self, tvars):
        # print(tvars)
        # print(tf.gradients(self.V[0], tvars))
        # print(tf.range(tf.shape(self.V)[-1]))
        # grads = tf.map_fn(lambda m : tf.gradients(self.V[m], tvars), tf.range(tf.shape(self.V)[-1]), tf.float32)
        #
        # V_unpacked = tf.unstack(grads)
        # for unit in V_unpacked:
        #     print(unit)
        return
