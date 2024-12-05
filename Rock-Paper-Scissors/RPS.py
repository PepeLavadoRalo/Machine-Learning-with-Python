# This function plays a rock-paper-scissors game by predicting the opponent's next move.
# It uses historical data, Markov Chains, and a pretraining process to improve its strategy.

import random

def player(prev_play, opponent_history=[], play_order={}):
    # Pretraining phase using a randomized version of the "quincy" player algorithm.
    # This generates a dataset of opponent's moves based on specific patterns.
    counter = 0
    if not opponent_history:
        while True:
            counter += 1
            # Take the last 5 moves from opponent's history.
            train_last_five = "".join(opponent_history[-5:])
            # Increment the count of this sequence in the play_order dictionary.
            play_order[train_last_five] = play_order.get(train_last_five, 0) + 1
            
            # Randomly generate the opponent's next move.
            choices = ["R", "R", "P", "P", "S", "S"]  # Adjusted distribution for variety.
            randomizer = bool(random.getrandbits(1))
            if randomizer:
                opponent_history.append(choices[counter % len(choices)])
            else:
                opponent_history.append(random.choice(choices))
            
            # Stop pretraining after 1000 iterations.
            if counter == 1000:
                break

    # Options for the possible next move: Rock (R), Paper (P), or Scissors (S).
    poss_next = ['R', 'P', 'S']
    
    # Mapping the opponent's moves to the ideal response for winning.
    ideal_response = {'P': 'S', 'R': 'P', 'S': 'R'}

    # If there is no previous play, default to "R" (Rock).
    if not prev_play:
        prev_play = 'R'
    
    # Default prediction for opponent's next move is "S" (Scissors).
    prediction = "S"

    # Add the opponent's last play to their history.
    opponent_history.append(prev_play)

    # Take the last 5 moves from the opponent's history.
    last_five = "".join(opponent_history[-5:])
    # Update the count of this sequence in the play_order dictionary.
    play_order[last_five] = play_order.get(last_five, 0) + 1

    # Create a list of possible sequences by adding each potential next move to the last 4 plays.
    potential_plays = []
    for v in poss_next:
        potential_plays.append("".join([*opponent_history[-4:], v]))

    # Check the frequencies of these potential sequences in the historical play_order.
    sub_order = {}
    for k in potential_plays:
        if k in play_order:
            sub_order[k] = play_order[k]

    # If historical data exists, predict the most likely next move by finding the sequence with the highest count.
    if sub_order:
        play_order[last_five] = play_order.get(last_five, 0) + 1
        prediction = max(sub_order, key=sub_order.get)[-1:]  # Last character indicates the predicted move.

    # Choose the response that beats the predicted move.
    guess = ideal_response[prediction]

    return guess
