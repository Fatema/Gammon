"""
This code is forked from https://github.com/fomorians/td-gammon, I made some modification so it allows the
creation of multiple instances of the network
"""

import os
import pickle
import numpy as np


class SubNet:
    def __init__(self, lamda=0.7, alpha=1, validation_interval=1000):
        self.validation_interval = validation_interval
        self.NUM = 0
        self.GAME_NUM = 0
        self.lamda = lamda
        self.alpha = alpha

    def set_network_name(self, name):
        self.STRATEGY = name

    def set_timestamp(self, timestamp):
        self.timestamp = timestamp

    def set_paths(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path + self.STRATEGY + '/latest/'
        self.previous_checkpoint_path = checkpoint_path + self.STRATEGY + '/previous/'
        self.test_checkpoint_path = checkpoint_path + self.STRATEGY + '/test/'

        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)

        if not os.path.exists(self.previous_checkpoint_path):
            os.makedirs(self.previous_checkpoint_path)

        if not os.path.exists(self.test_checkpoint_path):
            os.makedirs(self.test_checkpoint_path)

    def set_decay(self, lamda=0.7, alpha=1):
        # lambda decay
        self.lamda = lamda

        # learning rate decay
        self.alpha = alpha

    def set_nn(self, input=298, hidden=50, out=1, restore=False):
        # describe network size
        layer_size_input = input
        layer_size_hidden = hidden
        layer_size_output = out

        scales = [np.sqrt(6. / (layer_size_input + layer_size_hidden)),
                  np.sqrt(6. / (layer_size_output + layer_size_hidden))]

        self.weights = [scales[0] * np.random.randn(layer_size_input, layer_size_hidden),  # w_ih
                        scales[1] * np.random.randn(layer_size_hidden, layer_size_output),  # w_ho
                        np.zeros((layer_size_hidden, 1)),  # b_h
                        np.zeros((layer_size_output, 1))]  # b_o

        # the shape is based on the weights gradients shapes
        self.traces = [np.zeros((layer_size_input, layer_size_hidden, layer_size_output)),  # tw_iho
                       np.zeros((layer_size_hidden, layer_size_output)),  # tw_ho
                       np.zeros((layer_size_hidden, layer_size_output)),  # tb_ho
                       np.zeros((layer_size_output, 1))]  # tb_o

        if restore:
            self.restore()

    def sigmoid_activation(self, x, w, b):
        return np.matmul(x, w) + b

    def backprop(self, x, fpropOnly=True):
        w1, w2, b1, b2 = self.weights
        tw1, tw2, tb1, tb2 = self.traces

        # build network arch. (just 2 layers with sigmoid activation)
        prev_y = self.sigmoid_activation(x, w1, b1)
        V = self.sigmoid_activation(prev_y, w2, b2)

        if fpropOnly:
            return V

        # compute gradients
        db2_o = V * (1 - V)
        db1_ho = (prev_y * (1 - prev_y))[0][:, np.newaxis] * w2[:, :] * \
                 (db2_o)[0][np.newaxis, :]
        dw2_ho = prev_y[0][:, np.newaxis] * (db2_o)[0][np.newaxis, :]
        dw1_iho = x[0][:, np.newaxis, np.newaxis] * (prev_y * (1 - prev_y))[0][np.newaxis, :,
                                                    np.newaxis] * w2[np.newaxis, :, :] * \
                  (db2_o)[0][np.newaxis, np.newaxis, :]

        # update traces
        tw1 = self.lamda * tw1 + dw1_iho
        tw2 = self.lamda * tw2 + dw2_ho
        tb1 = self.lamda * tb1 + db1_ho
        tb2 = self.lamda * tw1 + db2_o

        self.traces = [tw1, tw2, tb1, tb2]

        # tw1 and tb2 dimensions are modified to it the weights dimensions (useful for more than 1 output)
        return V, [np.sum(tw1, axis=2), tw2, tb1, np.sum(tb2, axis=1)]

    def updateWeights(self, featsP, vN):
        # compute vals and grad
        vP, grad = self.backprop(featsP)

        scale = self.alpha * (vN - vP)
        for w, g in zip(self.weights, grad):
            w += scale * g

    def set_checkpoint(self):
        fid = open(self.checkpoint_path + "checkpoint-%d.bin" % self.NUM, 'w')
        pickle.dump([self.GAME_NUM, self.lamda, self.alpha, self.weights, self.traces], fid)
        fid.close()

    def set_previous_checkpoint(self):
        fid = open(self.previous_checkpoint_path + "checkpoint-%d.bin" % self.NUM, 'w')
        pickle.dump([self.GAME_NUM, self.lamda, self.alpha, self.weights, self.traces], fid)
        fid.close()

    def set_test_checkpoint(self):
        fid = open('{0}{1}/{2}'.format(self.test_checkpoint_path, self.timestamp, "checkpoint-%d.bin" % self.GAME_NUM),
                   'w')
        pickle.dump([self.GAME_NUM, self.lamda, self.alpha, self.weights, self.traces], fid)
        fid.close()

    def restore(self):
        try:
            self.GAME_NUM, self.lamda, self.alpha, self.weights, self.traces = pickle.load(
                open(self.checkpoint_path + 'checkpoint-%d.bin' % self.NUM, 'r'))
        except IOError:
            print("404 File not found!")

    def restore_previous(self):
        try:
            self.GAME_NUM, self.lamda, self.alpha, self.weights, self.traces = pickle.load(
                open(self.previous_checkpoint_path + 'checkpoint-%d.bin' % self.NUM, 'r'))
        except IOError:
            print("404 File not found!")

    def restore_test_checkpoint(self, timestamp, game_number):
        try:
            self.GAME_NUM, self.lamda, self.alpha, self.weights, self.traces = pickle.load(
                open('{0}{1}/{2}/'.format(self.test_checkpoint_path, timestamp, "checkpoint-%d.bin" % game_number),
                     'r'))
        except IOError:
            print("404 File not found!")

    def print_checkpoints(self):
        try:
            GAME_NUM, lamda, alpha, weights, traces = pickle.load(
                open(self.previous_checkpoint_path + 'checkpoint-%d.bin' % self.NUM, 'r'))
            print('previous checkpoint',
                  'game_number=' + GAME_NUM,
                  'lambda=' + lamda,
                  'alpha=' + alpha,
                  'weights=' + weights,
                  'traces=' + traces,
                  sep='\n')
        except IOError:
            print("404 File not found!")

        try:
            GAME_NUM, lamda, alpha, weights, traces = pickle.load(
                open(self.checkpoint_path + 'checkpoint-%d.bin' % self.NUM, 'r'))
            print('current checkpoint',
                  'game_number=' + GAME_NUM,
                  'lambda=' + lamda,
                  'alpha=' + alpha,
                  'weights=' + weights,
                  'traces=' + traces,
                  sep='\n')
        except IOError:
            print("404 File not found!")

    def get_output(self, x):
        return self.backprop(x, fpropOnly=True)

    def run_output(self, x, V_next):
        self.updateWeights(x, V_next)

    def update_model(self, x, winner):
        self.updateWeights(x, winner)

        if self.GAME_NUM == 1000:
            self.alpha = 0.1
        elif self.GAME_NUM == 100000:
            self.lamda = 0

        if self.GAME_NUM > 0 and self.GAME_NUM % self.validation_interval == 0:
            self.set_test_checkpoint()
            self.set_previous_checkpoint()

        self.GAME_NUM += 1

        # save weights
        self.set_checkpoint()
