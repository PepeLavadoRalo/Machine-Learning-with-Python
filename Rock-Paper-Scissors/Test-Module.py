import unittest
from RPS_game import play, mrugesh, abbey, quincy, kris  # Import the game mechanics and predefined bots.
from RPS import player  # Import the custom "player" function to test.

# Define the UnitTests class, which inherits from unittest.TestCase.
# This class contains unit tests for verifying the performance of the custom player.

class UnitTests(unittest.TestCase):
    print()  # Ensure there's a blank line for better output formatting.

    # Test the custom player against the "quincy" bot.
    def test_player_vs_quincy(self):
        print("Testing game against quincy...")
        # Check if the custom player achieves at least a 60% win rate over 1000 games.
        actual = play(player, quincy, 1000) >= 60
        self.assertTrue(
            actual,
            'Expected player to defeat quincy at least 60% of the time.'
        )

    # Test the custom player against the "abbey" bot.
    def test_player_vs_abbey(self):
        print("Testing game against abbey...")
        # Check if the custom player achieves at least a 60% win rate over 1000 games.
        actual = play(player, abbey, 1000) >= 60
        self.assertTrue(
            actual,
            'Expected player to defeat abbey at least 60% of the time.'
        )

    # Test the custom player against the "kris" bot.
    def test_player_vs_kris(self):
        print("Testing game against kris...")
        # Check if the custom player achieves at least a 60% win rate over 1000 games.
        actual = play(player, kris, 1000) >= 60
        self.assertTrue(
            actual,
            'Expected player to defeat kris at least 60% of the time.'
        )

    # Test the custom player against the "mrugesh" bot.
    def test_player_vs_mrugesh(self):
        print("Testing game against mrugesh...")
        # Check if the custom player achieves at least a 60% win rate over 1000 games.
        actual = play(player, mrugesh, 1000) >= 60
        self.assertTrue(
            actual,
            'Expected player to defeat mrugesh at least 60% of the time.'
        )


# The entry point for running the tests.
if __name__ == "__main__":
    unittest.main()
