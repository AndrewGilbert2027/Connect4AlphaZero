# Connect4 AlphaZero

This project is an implementation of the AlphaZero algorithm for the game of Connect4. It uses a PyTorch-based Convolutional Neural Network (CNN) for game state evaluation and a Monte Carlo Tree Search (MCTS) for move selection. The training process is accelerated through parallel self-play data generation.

## Project Structure

-   `C4.py`: The Connect4 game engine.
-   `Model.py`: The `Connect4CNN` neural network architecture.
-   `MCTS.py`: The Monte Carlo Tree Search implementation.
-   `train.py`: The main script for training the model via self-play.
-   `models/`: Directory where trained models are saved.

## Setup and Installation

Follow these steps to set up the project environment and run the training script.

### 1. Clone the Repository

First, clone the project to your local machine.

```bash
git clone <repository-url>
cd Connect4AlphaZero
```

### 2. Create a Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies.

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

Install the required Python packages using pip.

```bash
pip install torch numpy tqdm
```

## How to Run

To start the training process, run the `train.py` script. The script will begin generating self-play games in parallel, collecting training data, and training the neural network.

```bash
python train.py
```

The script is configured by default to run with parallel workers. You can adjust training parameters like the number of epochs, games per epoch, and batch size directly in the `if __name__ == "__main__":` block of `train.py`.

Trained models will be saved in the `models/` directory after each epoch.

## Areas for Improvement

While this implementation is a solid foundation, there are several areas where it could be enhanced:

1.  **C++ Game Engine with Python Bindings**: The current game logic is in Python. For a significant performance boost, the Connect4 game engine could be rewritten in C++ and exposed to Python using a library like `pybind11`. This would dramatically speed up the MCTS simulations, which are the primary bottleneck during data generation.

2.  **Elo-based Model Evaluation**: The current training loop saves the model after each epoch but doesn't evaluate its strength. An evaluation pipeline could be added where the new model plays a series of games against the previous best model. The new model would only be accepted as the "best" if it wins by a certain margin (e.g., an Elo rating increase), preventing the model from getting worse over time.

3.  **Hyperparameter Optimization and Learning Rate Scheduling**: The learning rate and MCTS exploration factor (`puct`) are currently static or use a simple decay schedule. Implementing more advanced learning rate scheduling (e.g., cyclical or one-cycle) and using a hyperparameter optimization framework (like Optuna or Ray Tune) could lead to faster convergence and a stronger final model.