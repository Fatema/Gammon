import os

import visdom

from backgammon.agents.ai_agent import TDAgent
from backgammon.agents.random_agent import RandomAgent
from backgammon.game import Game

from modnet import Modnet
from modnet_hybrid import ModnetHybrid
from mono_nn import MonoNN

import numpy as np

model_path = os.environ.get('MODEL_PATH', 'models/')
summary_path = os.environ.get('SUMMARY_PATH', 'summaries/')
checkpoint_path = os.environ.get('CHECKPOINT_PATH', 'checkpoints/')

previous_mod = Modnet(model_path, summary_path, checkpoint_path, restore=True)
previous_mod_hybrid = ModnetHybrid(model_path, summary_path, checkpoint_path, restore=True)
previous_mono = MonoNN(model_path, summary_path, checkpoint_path, restore=True)

vis = visdom.Visdom(server='ncc1.clients.dur.ac.uk',port=12345)

def run_games(players, episodes=100, draw=False):
    winners = [0, 0]
    winners_total = 1
    for episode in range(episodes):
        game = Game.new(layout=Game.LAYOUT)

        winner = game.play(players, draw=draw)
        winners[not winner] += 1

        winners_total = sum(winners)
        print("[Test %d] %s (%s) vs %s (%s) %d:%d of %d games (%.2f%%)" % (episode,
                                                                           players[0].player, players[0].player,
                                                                           players[1].player, players[1].player,
                                                                           winners[0], winners[1], winners_total,
                                                                           (winners[0] / winners_total) * 100.0))
    return (winners[0] / winners_total) * 100.0


def test_random(model, episodes=100, draw=False):
    players = [TDAgent(Game.TOKENS[0], model), RandomAgent(Game.TOKENS[1])]
    return run_games(players, episodes, draw)


def test_self(model, episodes=100, draw=False):
    if isinstance(model, Modnet):
        previous_model = previous_mod
    elif isinstance(model, ModnetHybrid):
        previous_model = previous_mod_hybrid
    else:
        previous_model = previous_mono

    previous_model.restore_previous()

    players = [TDAgent(Game.TOKENS[0], model), TDAgent(Game.TOKENS[1], previous_model)]
    run_games(players, episodes, draw)


def test_all_random(model, timestamp=0, max_checkpoint=500001, episodes=500, draw=False):
    vis.line(X=np.array([0]), Y=np.array([[np.nan]]), win='random')
    for i in range(1, max_checkpoint, 1000):
        if i == 1:
            model.restore_test_checkpoint(timestamp, i - 1)
        else:
            model.restore_test_checkpoint(timestamp, i)

        win_perc = test_random(model, episodes=episodes, draw=draw)

        # plot metrics
        vis.line(X=np.array([i]), Y=np.array([[
            win_perc,
        ]]), win='random', opts=dict(title='win', xlabel='checkpoint', ylabel='win rate', ytype='log', legend=[
            'random'
        ]), update='append')


def test_all_best(model, timestamp=0, max_checkpoint=500001, episodes=500, draw=False):
    vis.line(X=np.array([0]), Y=np.array([[np.nan]]), win='best')
    if isinstance(model, Modnet):
        previous_model = previous_mod
    elif isinstance(model, ModnetHybrid):
        previous_model = previous_mod_hybrid
    else:
        previous_model = previous_mono

    #restore best checkpoint from this timestamp
    model.restore_test_checkpoint(timestamp, max_checkpoint)

    for i in range(1, max_checkpoint, 1000):
        if i == 1:
            previous_model.restore_test_checkpoint(timestamp, i - 1)
        else:
            previous_model.restore_test_checkpoint(timestamp, i)

        players = [TDAgent(Game.TOKENS[0], model), TDAgent(Game.TOKENS[1], previous_model)]
        win_perc = run_games(players, episodes, draw)
        # plot metrics
        vis.line(X=np.array([i]), Y=np.array([[
            win_perc,
        ]]), win='best', opts=dict(title='win', xlabel='checkpoint', ylabel='win rate', ytype='log', legend=[
            'self'
        ]), update='append')

