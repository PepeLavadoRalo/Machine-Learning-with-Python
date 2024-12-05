# This is the entry point file for the Rock-Paper-Scissors project.
# Start by reading README.md for an overview of the project.

from RPS_game import play, mrugesh, abbey, quincy, kris, human, random_player
from RPS import player  # Import the custom "player" function you will develop.
from unittest import main  # Import the unit test framework.

# Test the custom "player" function against the predefined bots.
# Each bot has its own strategy, as explained in RPS_game.py.
# The "play" function simulates 1000 rounds between the custom player and the bot.

# Test against the "quincy" bot (predictable sequence strategy).
play(player, quincy, 1000)

# Test against the "abbey" bot (Markov chain prediction strategy).
play(player, abbey, 1000)

# Test against the "kris" bot (always counters the opponent's last move).
play(player, kris, 1000)

# Test against the "mrugesh" bot (most frequent move prediction strategy).
play(player, mrugesh, 1000)

# Uncomment the line below to play interactively against a bot.
# The "human" function allows you to input your own moves interactively.
# For example, play 20 rounds against the "abbey" bot with verbose output.
# play(human, abbey, 20, verbose=True)

# Uncomment the line below to play against a randomly playing bot.
# This can be useful for testing how well the custom player handles randomness.
# play(human, random_player, 1000)

# Uncomment the line below to run unit tests automatically.
# Unit tests are defined in the "test_module.py" file.
# main(module='test_module', exit=False)

