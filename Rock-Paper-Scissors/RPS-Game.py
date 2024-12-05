# DO NOT MODIFY THIS FILE
# This script sets up a rock-paper-scissors game simulation with multiple players and strategies.

import random

def play(player1, player2, num_games, verbose=False):
    """
    Simulates a game between two players for a specified number of games.
    
    Args:
    - player1, player2: Functions representing the players.
    - num_games: Number of rounds to play.
    - verbose: If True, prints detailed results of each round.
    
    Returns:
    - Player 1's win rate as a percentage.
    """
    p1_prev_play = ""  # Tracks player 1's last move.
    p2_prev_play = ""  # Tracks player 2's last move.
    results = {"p1": 0, "p2": 0, "tie": 0}  # Keeps track of wins, losses, and ties.

    for _ in range(num_games):
        # Get moves from both players.
        p1_play = player1(p2_prev_play)
        p2_play = player2(p1_prev_play)

        # Determine the outcome of the round.
        if p1_play == p2_play:
            results["tie"] += 1
            winner = "Tie."
        elif (p1_play == "P" and p2_play == "R") or \
             (p1_play == "R" and p2_play == "S") or \
             (p1_play == "S" and p2_play == "P"):
            results["p1"] += 1
            winner = "Player 1 wins."
        else:
            results["p2"] += 1
            winner = "Player 2 wins."

        if verbose:
            # Print details of the round if verbose mode is enabled.
            print("Player 1:", p1_play, "| Player 2:", p2_play)
            print(winner)
            print()

        # Update previous plays for the next round.
        p1_prev_play = p1_play
        p2_prev_play = p2_play

    games_won = results['p2'] + results['p1']  # Total games with a winner.

    # Calculate Player 1's win rate.
    win_rate = (results['p1'] / games_won * 100) if games_won > 0 else 0

    # Print final results and win rate.
    print("Final results:", results)
    print(f"Player 1 win rate: {win_rate}%")

    return win_rate


# "Quincy" strategy: Cycles through a predefined sequence of moves.
def quincy(prev_play, counter=[0]):
    counter[0] += 1  # Tracks the current step in the sequence.
    choices = ["R", "R", "P", "P", "S"]  # Predefined sequence.
    return choices[counter[0] % len(choices)]


# "Mrugesh" strategy: Predicts the opponent's most frequent move in their last 10 plays.
def mrugesh(prev_opponent_play, opponent_history=[]):
    opponent_history.append(prev_opponent_play)  # Track opponent's plays.
    last_ten = opponent_history[-10:]  # Consider the last 10 moves.
    most_frequent = max(set(last_ten), key=last_ten.count)  # Find the most frequent move.

    if most_frequent == '':
        most_frequent = "S"  # Default move if no data available.

    ideal_response = {'P': 'S', 'R': 'P', 'S': 'R'}  # Response to beat the opponent's move.
    return ideal_response[most_frequent]


# "Kris" strategy: Always counters the opponent's last move.
def kris(prev_opponent_play):
    if prev_opponent_play == '':
        prev_opponent_play = "R"  # Default to Rock if no previous move.
    ideal_response = {'P': 'S', 'R': 'P', 'S': 'R'}
    return ideal_response[prev_opponent_play]


# "Abbey" strategy: Uses a Markov Chain to predict the opponent's next move based on their last two moves.
def abbey(prev_opponent_play, opponent_history=[], play_order=[{
    "RR": 0, "RP": 0, "RS": 0,
    "PR": 0, "PP": 0, "PS": 0,
    "SR": 0, "SP": 0, "SS": 0,
}]):
    if not prev_opponent_play:
        prev_opponent_play = 'R'  # Default to Rock if no previous move.
    opponent_history.append(prev_opponent_play)  # Track opponent's moves.

    # Track the last two moves in a sequence.
    last_two = "".join(opponent_history[-2:])
    if len(last_two) == 2:
        play_order[0][last_two] += 1  # Increment the count of this sequence.

    # Generate potential plays by adding a possible next move.
    potential_plays = [
        prev_opponent_play + "R",
        prev_opponent_play + "P",
        prev_opponent_play + "S",
    ]

    # Subset of the order dictionary containing potential sequences.
    sub_order = {k: play_order[0][k] for k in potential_plays if k in play_order[0]}

    # Predict the opponent's next move based on historical frequency.
    prediction = max(sub_order, key=sub_order.get)[-1:]

    # Respond with the move that beats the predicted move.
    ideal_response = {'P': 'S', 'R': 'P', 'S': 'R'}
    return ideal_response[prediction]


# "Human" strategy: Prompts the user to input their move.
def human(prev_opponent_play):
    play = ""
    while play not in ['R', 'P', 'S']:  # Ensure valid input.
        play = input("[R]ock, [P]aper, [S]cissors? ").upper()
        print(play)
    return play


# "Random" strategy: Chooses a move randomly each time.
def random_player(prev_opponent_play):
    return random.choice(['R', 'P', 'S'])
