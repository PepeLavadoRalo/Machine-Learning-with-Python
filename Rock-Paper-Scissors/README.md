# Rock-Paper-Scissors AI Player

This project implements an AI player for the classic Rock-Paper-Scissors (RPS) game. The AI learns from its opponent's moves using a predictive algorithm based on historical patterns.
The goal is to consistently outperform predefined bots that use specific strategies.

## Features
- Custom AI (`player`) that uses a Markov Chain model and pretraining to predict opponent moves.
- Predefined opponent bots with varying strategies:
  - `quincy`: Cycles through a predetermined sequence.
  - `mrugesh`: Adapts based on the opponent's most frequent moves.
  - `kris`: Reacts directly to the opponent's last move.
  - `abbey`: Uses a two-move pattern to predict and counter.
- Unit tests to verify the AI's performance against all predefined bots.
- Options for human vs. bot or human vs. random player matches.

## File Structure

### 1. **`RPS.py`**
Contains the main implementation of the custom AI player:
- Tracks opponent move history and predicts future moves using a Markov Chain-based algorithm.
- Pretrained with a randomized version of the `quincy` bot's sequence for better performance.
- Selects the optimal move to counter the predicted opponent's next move.

### 2. **`RPS_game.py`**
Implements the game logic for RPS:
- Simulates matches between two players (AI vs. bot or bot vs. bot).
- Evaluates the win rate for the custom AI over a specified number of games.
- Includes predefined bots (`quincy`, `mrugesh`, `kris`, `abbey`, and a `random_player`) with unique strategies.

### 3. **`Main.py`**
Entry point for running the project:
- Tests the custom AI against the predefined bots for 1000 games each.
- Can optionally run interactive matches against a bot or a random player.
- Includes an option to run unit tests automatically.

### 4. **`Test-Module.py`**
Unit test suite to evaluate the AI:
- Ensures the custom AI achieves at least a 60% win rate against all predefined bots over 1000 games.
- Uses Python's `unittest` framework.

---

## Getting Started

### Prerequisites
- Python 3.7 or higher.
- Install required packages (if any).

### Running the AI
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
  ```
2. Run the predefined matches:
```bash
python Main.py
```
## Instructions for Customization

3. Uncomment lines in `Main.py` to:
- Play interactively against a bot (`human` vs. `abbey`).
- Play against a bot with random moves.

## How the AI Works

The AI (`player` in `RPS.py`) uses the following techniques:
1. **Opponent History**: Tracks the opponent's last five moves to identify patterns.
2. **Markov Chain Model**: Creates a probability distribution of possible next moves based on historical sequences.
3. **Pretraining**: Simulates 1000 moves against a randomized version of the `quincy` bot to populate initial data.
4. **Optimal Counter**: Selects the move that would most likely win against the opponent's predicted next move.

## Testing

Run the unit tests to ensure the AI meets performance expectations:
```bash
python Test-Module.py
```
Each test simulates 1000 games against a predefined bot.
The AI must achieve at least a 60% win rate for the test to pass.
## Example Output
Simulation Results
When running Main.py, you might see output like this:

```yaml
Testing against quincy...
Final results: {'p1': 650, 'p2': 300, 'tie': 50}
Player 1 win rate: 68.42%

Testing against abbey...
Final results: {'p1': 620, 'p2': 340, 'tie': 40}
Player 1 win rate: 64.63%
...
```
## Unit Test Output
Running Test-Module.py produces:

```python
Testing game against quincy...
Testing game against abbey...
Testing game against kris...
Testing game against mrugesh...

----------------------------------------------------------------------
Ran 4 tests in 0.5s

OK
```
## Future Improvements
Experiment with more complex predictive models (e.g., neural networks).

Incorporate reinforcement learning for self-improving strategies.

Add additional opponent bots with more advanced strategies.

Extend the interactive game to include score tracking and enhanced UI.






