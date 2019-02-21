import os

from backgammon.agents.ai_agent import TDAgent
from backgammon.agents.random_agent import RandomAgent
from backgammon.game import Game

from modnet import Modnet
from mono_nn import MonoNN

model_path = os.environ.get('MODEL_PATH', 'models/')
summary_path = os.environ.get('SUMMARY_PATH', 'summaries/')
checkpoint_path = os.environ.get('CHECKPOINT_PATH', 'checkpoints/')

previous_mod = Modnet(model_path, summary_path, checkpoint_path, restore=True)
previous_mono = MonoNN(model_path, summary_path, checkpoint_path, restore=True)


def test_random(model, episodes=1000, draw=False):
    players = [TDAgent(Game.TOKENS[0], model), RandomAgent(Game.TOKENS[1])]
    winners = [0, 0]
    for episode in range(episodes):
        game = Game.new()

        winner = 1 - game.play(players, draw=draw)
        winners[winner] += 1

        winners_total = sum(winners)
        print("[Episode %d] %s (%s) vs %s (%s) %d:%d of %d games (%.2f%%)" % (episode,
                                                                              players[0].player, players[0].player,
                                                                              players[1].player, players[1].player,
                                                                              winners[0], winners[1], winners_total,
                                                                              (winners[0] / winners_total) * 100.0))


def test_self(model, episodes=1000, draw=False):
    if isinstance(model, Modnet):
        previous_model = previous_mod
    else:
        previous_model = previous_mono

    previous_model.restore_previous()

    players = [TDAgent(Game.TOKENS[0], model), TDAgent(Game.TOKENS[1], previous_model)]
    winners = [0, 0]
    for episode in range(episodes):
        game = Game.new()

        winner = 1 - game.play(players, draw=draw)
        winners[winner] += 1

        winners_total = sum(winners)
        print("[Episode %d] %s (%s) vs %s (%s) %d:%d of %d games (%.2f%%)" % (episode,
                                                                              players[0].player, players[0].player,
                                                                              players[1].player, players[1].player,
                                                                              winners[0], winners[1], winners_total,
                                                                              (winners[0] / winners_total) * 100.0))
    model.set_previous_checkpoint()
