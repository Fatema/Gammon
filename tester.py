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


def test_random(model, episodes=100, draw=False):
    players = [TDAgent(Game.TOKENS[0], model), RandomAgent(Game.TOKENS[1])]
    winners = [0, 0]
    for episode in range(episodes):
        game = Game.new()

        winner = game.play(players, draw=draw)
        winners[not winner] += 1

        winners_total = sum(winners)
    print("[Test %d] %s (%s) vs %s (%s) %d:%d of %d games (%.2f%%)" % (episode,
                       players[0].player, players[0].player,
                       players[1].player, players[1].player,
                       winners[0], winners[1], winners_total,
                       (winners[0] / winners_total) * 100.0))


def test_self(model, episodes=100, draw=False):
    if isinstance(model, Modnet):
        previous_model = previous_mod
    else:
        previous_model = previous_mono

    previous_model.restore_previous()
    model.print_checkpoints()

    players = [TDAgent(Game.TOKENS[0], model), TDAgent(Game.TOKENS[1], previous_model)]
    winners = [0, 0]
    for episode in range(episodes):
        game = Game.new()

        winner = game.play(players, draw=draw)
        winners[not winner] += 1

        winners_total = sum(winners)
    print("[Test %d] %s (%s) vs %s (%s) %d:%d of %d games (%.2f%%)" % (episode,
                       players[0].player, players[0].player,
                       players[1].player, players[1].player,
                       winners[0], winners[1], winners_total,
                       (winners[0] / winners_total) * 100.0))


def test_all(model, timestamp=1551266794, episodes=100, draw=False):
    if isinstance(model, Modnet):
        previous_model = previous_mod
    else:
        previous_model = previous_mono

    checkpoints = [0, 1007262, 1056536, 1106029, 1155408, 1204768, 1254983, 1304685, 1354754, 1404715, 143785, 1454107,
                   1503361, 1553289, 1603371, 1653415, 1703199, 1753204, 1803466, 1853649, 1903980, 193783, 1953852,
                   2004102, 2054550, 2104497, 2155002, 2205137, 2255598, 2305958, 2357094, 2407649, 2458034, 246874,
                   2508276, 2558758, 2609612, 2659913, 2710426, 2760713, 2811339, 2861871, 2912318, 2962568, 298095,
                   3013295, 3063723, 3114328, 3164879, 3215420, 3265807, 3316198, 3366800, 3417574, 3468047, 3518700,
                   353084, 3569135, 3619883, 3670421, 3721113, 3771879, 3822773, 3873310, 3924061, 3974811, 4025466,
                   403167, 4075817, 4126795, 4177558, 4228627, 4279449, 4330508, 4381442, 4432354, 4483368, 4534374,
                   4585400, 458976, 4636587, 4687604, 47291, 4738634, 4789686, 4840738, 4891411, 4942589, 4993811,
                   5044881, 5096362, 510646, 5147388, 5198588, 5249906, 5301102, 5352559, 5403960, 5456058, 5507442,
                   5558949, 560921, 5610596, 5661887, 5713636, 5765307, 5816659, 5868460, 5919546, 610890, 660579,
                   710082, 759346, 808600, 857994, 908384, 95117, 957823]
    sorted(checkpoints)

    for i in range(len(checkpoints)):
        model.restore_test_checkpoint(timestamp, checkpoints[i])
        test_random(model, episodes=episodes, draw=draw)

    for i in range(1, len(checkpoints)):
        previous_model.restore_test_checkpoint(timestamp, checkpoints[i - 1])
        model.restore_test_checkpoint(timestamp, checkpoints[i])

        players = [TDAgent(Game.TOKENS[0], model), TDAgent(Game.TOKENS[1], previous_model)]
        winners = [0, 0]

        for episode in range(episodes):
            game = Game.new()

            winner = game.play(players, draw=draw)
            winners[not winner] += 1

            winners_total = sum(winners)


        print("[Test %d] %s (%s) vs %s (%s) %d:%d of %d games (%.2f%%)" % (episode,
                           players[0].player, players[0].player,
                           players[1].player, players[1].player,
                           winners[0], winners[1], winners_total,
                           (winners[0] / winners_total) * 100.0))
