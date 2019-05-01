import os

import visdom

from backgammon.agents.ai_agent import TDAgent, PubevalAgent
from backgammon.agents.human_agent import HumanAgent
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

vis = visdom.Visdom(server='localhost', port=12345)

mono_pip = [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 44.00, 46.00, 33.00, 29.00, 32.00, 35.00, 37.00, 35.00, 38.00, 37.00,
            36.00, 39.00, 42.00, 34.00, 40.00, 47.00, 37.00, 42.00, 36.00, 42.00, 40.00, 47.00, 30.00, 33.00, 44.00,
            32.00,
            32.00, 38.00, 31.00, 35.00, 40.00, 34.00, 36.00, 53.00, 46.00, 47.00, 52.00, 41.00, 52.00, 52.00, 53.00,
            49.00,
            51.00, 47.00, 55.00, 52.00, 52.00, 48.00, 49.00, 57.00, 48.00, 51.00, 46.00, 52.00, 47.00, 54.00, 55.00,
            50.00,
            53.00, 53.00, 50.00, 40.00, 52.00, 54.00, 47.00, 53.00, 45.00, 50.00, 51.00, 59.00, 36.00, 54.00, 55.00,
            46.00,
            50.00, 46.00, 57.00, 52.00, 43.00, 53.00, 54.00, 50.00, 55.00, 49.00, 51.00, 54.00, 63.00, 52.00, 64.00,
            53.00,
            46.00, 53.00, 56.00, 50.00, 50.00, 55.00, 48.00, 56.00, 42.00, 50.00, 50.00, 56.00, 47.00, 49.00, 48.00,
            50.00,
            52.00, 33.00, 56.00, 50.00, 41.00, 49.00, 46.00]
mono_all = [15.00, 20.00, 19.00, 12.00, 19.00, 19.00, 16.00, 17.00, 12.00, 12.00, 14.00, 13.00, 14.00, 14.00, 16.00,
            8.00, 12.00, 12.00, 19.00, 16.00, 15.00, 17.00, 14.00, 21.00, 16.00, 9.00, 16.00, 14.00, 54.00, 54.00,
            47.00, 54.00, 45.00, 57.00, 45.00, 59.00, 52.00, 58.00, 56.00, 54.00, 50.00, 57.00, 59.00, 59.00, 47.00,
            57.00, 52.00, 53.00, 50.00, 42.00, 51.00, 50.00, 45.00, 51.00, 53.00, 62.00, 53.00, 56.00, 56.00, 55.00,
            52.00, 42.00, 60.00, 64.00, 56.00, 53.00, 59.00, 59.00, 55.00, 56.00, 58.00, 59.00, 61.00, 58.00, 57.00,
            64.00, 63.00, 59.00, 53.00, 61.00, 51.00, 61.00, 57.00, 54.00, 59.00, 67.00, 61.00, 59.00, 66.00, 69.00,
            54.00, 57.00, 60.00, 52.00, 55.00, 63.00, 63.00, 62.00, 59.00, 60.00, 66.00, 63.00, 63.00, 62.00, 58.00,
            58.00, 51.00, 58.00, 56.00, 52.00]
mono_non = [3.00, 5.00, 5.00, 5.00, 7.00, 4.00, 7.00, 6.00, 4.00, 7.00, 5.00, 53.00, 38.00, 49.00, 46.00, 37.00, 45.00,
            41.00, 40.00, 40.00, 46.00, 40.00, 48.00, 43.00, 36.00, 43.00, 38.00, 38.00, 39.00, 36.00, 42.00, 35.00,
            54.00, 36.00, 45.00, 38.00, 42.00, 40.00, 50.00, 34.00, 44.00, 41.00, 37.00, 39.00, 44.00, 47.00, 39.00,
            37.00, 48.00, 49.00, 37.00, 37.00, 40.00, 39.00, 43.00, 43.00, 43.00, 35.00, 46.00, 51.00, 39.00, 43.00,
            41.00, 39.00, 47.00, 36.00, 37.00, 47.00, 47.00, 43.00, 46.00, 44.00, 39.00, 43.00, 35.00, 42.00, 31.00,
            35.00, 44.00, 39.00, 37.00, 50.00, 45.00, 38.00, 37.00, 41.00, 50.00, 38.00, 42.00, 47.00, 33.00, 38.00,
            44.00, 38.00, 48.00, 49.00, 43.00, 41.00, 38.00, 41.00, 42.00, 33.00, 43.00, 56.00, 42.00, 48.00, 42.00,
            44.00, 40.00, 48.00, 42.00, 48.00, 45.00, 39.00, 51.00, 50.00, 60.00, 63.00, 52.00, 55.00, 58.00, 64.00,
            64.00, 48.00, 53.00, 57.00, 54.00, 67.00, 53.00, 57.00, 56.00, 63.00, 61.00, 55.00, 62.00, 62.00, 63.00,
            61.00, 58.00, 60.00, 55.00, 51.00, 60.00, 49.00, 58.00, 60.00, 54.00, 62.00, 53.00, 63.00, 56.00, 63.00,
            55.00, 56.00, 55.00, 57.00, 74.00, 59.00, 57.00, 66.00, 57.00, 67.00, 62.00, 58.00, 64.00, 61.00, 51.00,
            60.00, 66.00, 54.00, 62.00, 58.00, 66.00, 68.00, 57.00, 55.00, 59.00, 63.00, 59.00, 63.00, 68.00, 59.00,
            61.00, 62.00, 57.00, 58.00, 59.00, 47.00, 56.00, 56.00, 44.00, 56.00, 63.00, 50.00, 47.00, 58.00, 62.00,
            55.00, 45.00, 55.00, 57.00, 53.00, 51.00, 57.00, 43.00, 54.00, 50.00, 49.00, 54.00, 54.00, 55.00, 60.00,
            59.00, 49.00, 55.00, 50.00, 51.00, 60.00, 55.00, 49.00, 45.00, 50.00, 42.00, 51.00, 44.00, 58.00, 58.00,
            47.00, 58.00, 62.00, 43.00, 58.00, 54.00, 55.00, 55.00, 53.00, 54.00, 42.00, 51.00, 56.00, 49.00, 52.00,
            53.00, 52.00, 58.00, 53.00]
mod_all = [1.00, 7.00, 7.00, 5.00, 3.00, 6.00, 4.00, 1.00, 5.00, 4.00, 16.00, 24.00, 23.00, 14.00, 17.00, 14.00, 12.00,
           16.00, 9.00, 12.00, 20.00, 13.00, 4.00, 2.00, 0.00]
mod_pip = [3.00, 0.00, 1.00, 2.00, 13.00, 12.00, 13.00, 16.00, 19.00, 9.00, 8.00, 6.00, 22.00, 16.00, 3.00, 1.00, 1.00,
           0.00, 1.00, 3.00, 12.00, 17.00, 21.00, 21.00, 23.00, 20.00, 12.00, 27.00, 17.00, 22.00, 17.00, 19.00, 26.00,
           21.00, 16.00, 9.00, 21.00, 16.00, 15.00, 12.00, 15.00, 11.00, 13.00, 13.00, 9.00, 14.00, 10.00, 4.00, 8.00,
           6.00, 4.00, 1.00, 4.00, 3.00, 2.00, 1.00, 3.00, 2.00, 1.00, 0.00]
mod_non = [1.00, 1.00, 3.00, 9.00, 14.00, 17.00, 16.00, 14.00, 21.00, 21.00, 11.00, 19.00, 20.00, 16.00, 26.00, 21.00,
           16.00, 23.00, 21.00, 10.00, 9.00, 12.00, 15.00, 13.00, 17.00, 18.00, 10.00, 15.00, 13.00, 24.00, 24.00,
           27.00, 30.00, 23.00, 18.00, 21.00, 21.00, 19.00, 20.00, 24.00, 19.00, 29.00, 27.00, 31.00, 14.00, 24.00,
           28.00, 24.00, 28.00, 17.00, 27.00, 23.00, 15.00, 17.00, 24.00, 28.00, 23.00, 20.00, 36.00, 33.00, 27.00,
           23.00, 18.00, 19.00, 20.00, 27.00, 31.00, 23.00, 31.00, 30.00, 27.00, 31.00, 21.00, 33.00, 28.00, 28.00,
           14.00, 17.00, 18.00, 28.00, 30.00, 25.00, 30.00, 26.00, 20.00, 23.00, 23.00, 27.00, 23.00, 23.00, 26.00,
           17.00, 18.00, 20.00, 16.00, 28.00, 18.00, 19.00, 18.00, 27.00, 31.00, 24.00, 25.00, 29.00, 27.00, 33.00,
           24.00, 33.00, 28.00, 30.00, 23.00, 32.00, 22.00, 22.00, 26.00, 40.00, 28.00, 24.00, 33.00, 30.00, 35.00,
           40.00, 29.00, 24.00, 29.00, 32.00, 25.00, 25.00, 29.00, 32.00, 24.00, 33.00]
hybrid_all = [3.00, 2.00, 4.00, 13.00, 7.00, 7.00, 24.00, 22.00, 19.00, 9.00, 11.00, 18.00, 11.00, 16.00, 22.00, 25.00,
              20.00, 24.00, 24.00, 15.00, 6.00, 17.00, 16.00, 16.00, 9.00, 10.00, 25.00, 12.00, 14.00, 29.00, 14.00,
              6.00, 11.00, 10.00, 14.00, 18.00, 15.00, 9.00, 19.00, 8.00, 15.00, 10.00, 5.00, 2.00, 4.00, 5.00, 8.00,
              10.00, 6.00, 7.00, 10.00, 15.00, 11.00, 6.00]
hybrid_pip = [0.00, 2.00, 3.00, 7.00, 7.00, 22.00, 14.00, 23.00, 17.00, 9.00, 16.00, 14.00, 11.00, 19.00, 22.00, 25.00,
              18.00, 25.00, 29.00, 22.00, 20.00, 13.00, 16.00, 17.00, 25.00, 23.00, 32.00, 28.00, 14.00, 24.00, 18.00,
              20.00, 24.00, 23.00, 25.00, 29.00, 31.00, 25.00, 30.00, 24.00, 28.00, 25.00, 18.00, 27.00, 21.00, 24.00,
              26.00, 23.00, 28.00, 22.00, 21.00, 29.00, 18.00, 22.00, 26.00, 25.00, 26.00, 24.00, 29.00, 36.00, 22.00,
              17.00, 29.00, 25.00, 33.00, 15.00, 30.00, 29.00, 34.00, 33.00, 23.00, 30.00, 31.00, 28.00, 25.00, 23.00,
              28.00, 27.00, 37.00, 31.00, 29.00, 36.00, 32.00, 27.00, 28.00, 35.00, 34.00, 26.00, 19.00, 26.00, 29.00,
              32.00, 30.00, 30.00, 35.00, 39.00, 24.00, 31.00, 26.00, 26.00, 21.00, 24.00, 32.00, 28.00, 16.00, 25.00,
              29.00, 30.00, 30.00, 33.00, 36.00, 26.00, 31.00, 32.00, 27.00, 22.00, 38.00, 27.00, 36.00, 39.00, 33.00,
              30.00, 33.00, 36.00, 31.00, 24.00, 36.00, 34.00, 35.00, 27.00, 32.00, 16.00, 16.00, 30.00, 19.00, 25.00,
              38.00, 28.00, 26.00, 19.00, 26.00, 31.00, 27.00, 25.00, 27.00, 27.00, 29.00, 17.00, 19.00, 22.00, 19.00,
              21.00, 18.00, 26.00, 26.00, 33.00, 20.00, 12.00, 24.00, 23.00, 20.00, 29.00, 30.00, 27.00, 23.00, 34.00,
              24.00, 24.00, 27.00, 29.00, 41.00, 22.00, 34.00, 29.00, 31.00, 37.00, 29.00, 34.00, 24.00, 18.00, 15.00]
hybrid_non = [0.00, 4.00, 3.00, 10.00, 11.00, 8.00, 19.00, 19.00, 24.00, 17.00, 19.00, 22.00, 15.00, 22.00, 16.00, 9.00,
              11.00, 4.00, 8.00, 5.00, 5.00, 3.00, 9.00, 7.00, 15.00, 10.00, 20.00, 18.00, 15.00, 23.00, 12.00, 10.00,
              10.00, 8.00, 15.00, 9.00, 10.00, 14.00, 5.00, 9.00, 6.00, 6.00, 12.00, 10.00, 18.00, 24.00, 19.00, 17.00,
              18.00, 13.00, 16.00, 13.00, 16.00, 16.00, 21.00, 14.00, 22.00, 14.00, 15.00]

def create_plots():
    mnon = len(mono_non)
    dnon = len(mod_non)
    hnon = len(hybrid_non)

    mpip = len(mono_pip)
    dpip = len(mod_pip)
    hpip = len(hybrid_pip)

    mall = len(mono_all)
    dall = len(mod_all)
    hall = len(hybrid_all)

    vis.line(X=np.array([0]), Y=np.array([[np.nan,np.nan,np.nan]]), win='non')
    vis.line(X=np.array([0]), Y=np.array([[np.nan,np.nan,np.nan]]), win='pip')
    vis.line(X=np.array([0]), Y=np.array([[np.nan,np.nan,np.nan]]), win='all')

    vis.line(X=np.array([0]), Y=np.array([[np.nan]]), win='mono_non')
    vis.line(X=np.array([0]), Y=np.array([[np.nan]]), win='mono_pip')
    vis.line(X=np.array([0]), Y=np.array([[np.nan]]), win='mono_all')

    for i in range(mnon):
        # plot metrics
        vis.line(X=np.array([i*1000]), Y=np.array([[
            mono_non[i]
        ]]), win='mono_non', opts=dict(title='win', xlabel='checkpoint', ylabel='win rate', ytype='log', legend=[
            'monolithic'
        ]), update='append')

    for i in range(mpip):
        # plot metrics
        vis.line(X=np.array([i*1000]), Y=np.array([[
            mono_pip[i]
        ]]), win='mono_pip', opts=dict(title='win', xlabel='checkpoint', ylabel='win rate', ytype='log', legend=[
            'monolithic'
        ]), update='append')

    for i in range(mall):
        # plot metrics
        vis.line(X=np.array([i*1000]), Y=np.array([[
            mono_all[i]
        ]]), win='mono_all', opts=dict(title='win', xlabel='checkpoint', ylabel='win rate', ytype='log', legend=[
            'monolithic'
        ]), update='append')

    vis.line(X=np.array([0]), Y=np.array([[np.nan]]), win='mod_non')
    vis.line(X=np.array([0]), Y=np.array([[np.nan]]), win='mod_pip')
    vis.line(X=np.array([0]), Y=np.array([[np.nan]]), win='mod_all')

    for i in range(dnon):
        # plot metrics
        vis.line(X=np.array([i*1000]), Y=np.array([[
            mod_non[i]
        ]]), win='mod_non', opts=dict(title='win', xlabel='checkpoint', ylabel='win rate', ytype='log', legend=[
            'seperate modular'
        ]), update='append')

    for i in range(dpip):
        # plot metrics
        vis.line(X=np.array([i*1000]), Y=np.array([[
            mod_pip[i]
        ]]), win='mod_pip', opts=dict(title='win', xlabel='checkpoint', ylabel='win rate', ytype='log', legend=[
            'seperate modular'
        ]), update='append')

    for i in range(dall):
        # plot metrics
        vis.line(X=np.array([i*1000]), Y=np.array([[
            mod_all[i]
        ]]), win='mod_all', opts=dict(title='win', xlabel='checkpoint', ylabel='win rate', ytype='log', legend=[
            'seperate modular'
        ]), update='append')

    vis.line(X=np.array([0]), Y=np.array([[np.nan]]), win='hybrid_non')
    vis.line(X=np.array([0]), Y=np.array([[np.nan]]), win='hybrid_pip')
    vis.line(X=np.array([0]), Y=np.array([[np.nan]]), win='hybrid_all')

    for i in range(hnon):
        # plot metrics
        vis.line(X=np.array([i*1000]), Y=np.array([[
            hybrid_non[i]
        ]]), win='hybrid_non', opts=dict(title='win', xlabel='checkpoint', ylabel='win rate', ytype='log', legend=[
            'hybrid modular'
        ]), update='append')

    for i in range(hpip):
        # plot metrics
        vis.line(X=np.array([i*1000]), Y=np.array([[
            hybrid_pip[i]
        ]]), win='hybrid_pip', opts=dict(title='win', xlabel='checkpoint', ylabel='win rate', ytype='log', legend=[
            'hybrid modular'
        ]), update='append')

    for i in range(hall):
        # plot metrics
        vis.line(X=np.array([i*1000]), Y=np.array([[
            hybrid_all[i]
        ]]), win='hybrid_all', opts=dict(title='win', xlabel='checkpoint', ylabel='win rate', ytype='log', legend=[
            'hybrid modular'
        ]), update='append')

    for i in range(min(mnon, hnon, dnon,mpip, hpip, dpip,mall, hall, dall)):
        # plot metrics
        vis.line(X=np.array([i*1000]), Y=np.array([[
            mono_non[i],
            mod_non[i],
            hybrid_non[i]
        ]]), win='non', opts=dict(title='win', xlabel='checkpoint', ylabel='win rate', ytype='log', legend=[
            'monolithic',
            'seperate modular',
            'hybrid modular'
        ]), update='append')

    # for i in range(min(mpip, hpip, dpip)):
        # plot metrics
        vis.line(X=np.array([i*1000]), Y=np.array([[
            mono_pip[i],
            mod_pip[i],
            hybrid_pip[i]
        ]]), win='pip', opts=dict(title='win', xlabel='checkpoint', ylabel='win rate', ytype='log', legend=[
            'monolithic',
            'seperate modular',
            'hybrid modular'
        ]), update='append')

    # for i in range(min(mall, hall, dall)):
        # plot metrics
        vis.line(X=np.array([i*1000]), Y=np.array([[
            mono_all[i],
            mod_all[i],
            hybrid_all[i]
        ]]), win='all', opts=dict(title='win', xlabel='checkpoint', ylabel='win rate', ytype='log', legend=[
            'monolithic',
            'seperate modular',
            'hybrid modular'
        ]), update='append')


def manual_test_gnubg(model):
    game = Game.new(layout=Game.LAYOUT)
    players = [TDAgent(Game.TOKENS[0], model), HumanAgent(Game.TOKENS[1])]
    player_num = int(input('second player num'))
    turns = 0
    while not game.is_over():
        turns += 1
        nodups = False
        roll = tuple(map(int, input('dice roll').split(',')))
        player_num = (player_num + 1) % 2
        if player_num:
            nodups = True
            game.reverse()
        game.take_turn(players[player_num], roll, draw=False, nodups=nodups)
        if player_num:
            game.reverse()
    winner = game.winner()
    return winner


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


def test_pubeval(model, episodes=100, draw=False):
    players = [TDAgent(Game.TOKENS[0], model), PubevalAgent(Game.TOKENS[1])]
    # players = [RandomAgent(Game.TOKENS[0]), PubevalAgent(Game.TOKENS[1])]
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
    for i in range(1, max_checkpoint, 5000):
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

    # restore best checkpoint from this timestamp
    model.restore_test_checkpoint(timestamp, max_checkpoint)

    for i in range(1, max_checkpoint, 5000):
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
