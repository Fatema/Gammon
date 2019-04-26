"""
This code is forked from https://github.com/fomorians/td-gammon, I made some modification so it allows the
creation of multiple instances of the network
"""

import os
import pickle
import numpy as np
import visdom

vis = visdom.Visdom(server='localhost', port=12345)


class SubNet:
    def __init__(self, lamda=0.7, alpha=1, validation_interval=1000):
        self.validation_interval = validation_interval
        self.NUM = 0
        self.GAME_NUM = 0
        self.lamda = lamda
        self.alpha = alpha
        self.global_step = 0
        self.loss_avg = 0.0
        self.delta_avg = 0.0

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

        self.weights = [(scales[0] * np.random.randn(layer_size_hidden, layer_size_input)),  # w_hi
                        (scales[1] * np.random.randn(layer_size_output, layer_size_hidden)),  # w_oh
                        np.zeros((layer_size_hidden, 1)),  # b_h
                        np.zeros((layer_size_output, 1))]  # b_o

        # the shape is based on the weights gradients shapes
        self.traces = [np.zeros((layer_size_hidden, layer_size_input)),  # tw_iho
                       np.zeros((layer_size_output, layer_size_hidden)),  # tw_ho
                       np.zeros((layer_size_hidden, 1)),  # tb_ho
                       np.zeros((layer_size_output, 1))]  # tb_o

        # vis.line(X=np.array([0]), Y=np.array([[np.nan]]), win='loss')
        # vis.line(X=np.array([0]), Y=np.array([[np.nan]]), win='loss_avg')
        # vis.line(X=np.array([0]), Y=np.array([[np.nan]]), win='delta')
        # vis.line(X=np.array([0]), Y=np.array([[np.nan]]), win='delta_avg')

        if restore:
            self.restore()

    def sigmoid_activation(self, x, w, b):
        return 1 / (1 + np.exp(-(w.dot(x) + b)))

    def backprop(self, x, fpropOnly=False):
        w1, w2, b1, b2 = self.weights
        tw1, tw2, tb1, tb2 = self.traces

        # build network arch. (just 2 layers with sigmoid activation)
        prev_y = self.sigmoid_activation(x.T, w1, b1)
        V = self.sigmoid_activation(prev_y, w2, b2)

        if fpropOnly:
            return V

        # compute gradients
        db2_o = V * (1 - V)
        db1_ho = w2.T * db2_o * prev_y * (1 - prev_y)
        dw2_ho = db2_o * prev_y.T
        dw1_iho = db1_ho * x

        # print(db2_o.shape, db1_ho.shape, dw2_ho.shape, dw1_iho.shape)

        # update traces
        tw1 = self.lamda * tw1 + dw1_iho
        tw2 = self.lamda * tw2 + dw2_ho
        tb1 = self.lamda * tb1 + db1_ho
        tb2 = self.lamda * tb2 + db2_o

        # print(x.shape, w2.shape, prev_y.shape, V.shape)

        self.traces = [tw1, tw2, tb1, tb2]

        # print(tw1.shape, tw2.shape, tb1.shape, tb2.shape, V.shape)

        # tw1 and tb2 dimensions are modified to it the weights dimensions (useful for more than 1 output)
        return V, [tw1, tw2, tb1, tb2]

    def updateWeights(self, featsP, vN):
        self.global_step += 1

        # compute vals and grad
        vP, grad = self.backprop(featsP)

        delta = np.sum(vN - vP)
        self.delta_avg += delta

        loss = np.mean(np.square(delta))
        self.loss_avg += loss

        scale = self.alpha * delta

        # vis.line(X=np.array([self.global_step]), Y=np.array([[
        #     delta
        # ]]), win='delta', opts=dict(title='win', xlabel='step', ylabel='delta', ytype='log', legend=[
        #     'delta',
        # ]), update='append')
        # vis.line(X=np.array([self.global_step]), Y=np.array([[
        #     self.delta_avg / self.global_step
        # ]]), win='delta_avg', opts=dict(title='win', xlabel='step', ylabel='delta', ytype='log', legend=[
        #     'delta-avg',
        # ]), update='append')
        #
        # vis.line(X=np.array([self.global_step]), Y=np.array([[
        #     loss
        # ]]), win='loss', opts=dict(title='win', xlabel='step', ylabel='loss', ytype='log', legend=[
        #     'loss',
        # ]), update='append')
        # vis.line(X=np.array([self.global_step]), Y=np.array([[
        #     self.loss_avg / self.global_step
        # ]]), win='loss_avg', opts=dict(title='win', xlabel='step', ylabel='loss', ytype='log', legend=[
        #     'loss-avg',
        # ]), update='append')

        w1, w2, b1, b2 = self.weights
        tw1, tw2, tb1, tb2 = grad

        w1 += scale * tw1
        w2 += scale * w2
        b1 += scale * tb1
        b2 += scale * tb2

        self.weights = [w1, w2, b1, b2]

    def set_checkpoint(self):
        fid = open(self.checkpoint_path + "checkpoint-%d.bin" % self.NUM, 'wb')
        pickle.dump([self.GAME_NUM, self.global_step, self.lamda, self.alpha, self.weights, self.traces], fid)
        fid.close()

    def set_previous_checkpoint(self):
        fid = open(self.previous_checkpoint_path + "checkpoint-%d.bin" % self.NUM, 'wb')
        pickle.dump([self.GAME_NUM, self.global_step, self.lamda, self.alpha, self.weights, self.traces], fid)
        fid.close()

    def set_test_checkpoint(self):
        if not os.path.exists(self.test_checkpoint_path + str(self.timestamp)):
            os.makedirs(self.test_checkpoint_path + str(self.timestamp))
        fid = open('{0}{1}/{2}'.format(self.test_checkpoint_path, self.timestamp, "checkpoint-%d.bin" % self.GAME_NUM),
                   'wb')
        pickle.dump([self.GAME_NUM, self.global_step, self.lamda, self.alpha, self.weights, self.traces], fid)
        fid.close()

    def restore(self):
        try:
            self.GAME_NUM, self.global_step, self.lamda, self.alpha, self.weights, self.traces = pickle.load(
                open(self.checkpoint_path + 'checkpoint-%d.bin' % self.NUM, 'rb'))
        except IOError:
            print("404 File not found!")

    def restore_previous(self):
        try:
            self.GAME_NUM, self.global_step, self.lamda, self.alpha, self.weights, self.traces = pickle.load(
                open(self.previous_checkpoint_path + 'checkpoint-%d.bin' % self.NUM, 'rb'))
        except IOError:
            print("404 File not found!")

    def restore_test_checkpoint(self, timestamp, game_number):
        try:
            self.GAME_NUM, self.global_step, self.lamda, self.alpha, self.weights, self.traces = pickle.load(
                open('{0}{1}/{2}/'.format(self.test_checkpoint_path, timestamp, "checkpoint-%d.bin" % game_number),
                     'rb'))
        except IOError:
            print("404 File not found!")

    def print_checkpoints(self):
        try:
            GAME_NUM, global_step, lamda, alpha, weights, traces = pickle.load(
                open(self.previous_checkpoint_path + 'checkpoint-%d.bin' % self.NUM, 'r'))
            print('previous checkpoint',
                  'game_number=' + GAME_NUM,
                  'global_steps_number=', global_step,
                  'lambda=' + lamda,
                  'alpha=' + alpha,
                  'weights=' + weights,
                  'traces=' + traces,
                  sep='\n')
        except IOError:
            print("404 previous not found!")

        try:
            GAME_NUM, global_step, lamda, alpha, weights, traces = pickle.load(
                open(self.checkpoint_path + 'checkpoint-%d.bin' % self.NUM, 'r'))
            print('current checkpoint',
                  'game_number=' + GAME_NUM,
                  'global_steps_number=', global_step,
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

        if self.GAME_NUM % self.validation_interval == 0:
            self.set_test_checkpoint()
            self.set_previous_checkpoint()

        self.GAME_NUM += 1

        # save weights
        if self.GAME_NUM > 0 and self.GAME_NUM % 500 == 0:
            self.set_checkpoint()
