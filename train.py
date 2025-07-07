import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import os
import time
import concurrent.futures
from tqdm import tqdm
from Model import Connect4CNN
from MCTS import MCTS_Deep
from C4 import C4

# Worker function to run a single game in a separate process
def play_game(game_id, game_class, num_simulations, temperature, bootstrap):
    """Self-contained function to play a complete game in a worker process."""
    # Each worker creates its own game and model instance
    game = game_class()
    model = Connect4CNN()
    
    # Load the model state from the temporary file
    try:
        model.load("temp_model.pth")
    except Exception as e:
        print(f"Worker {game_id}: Could not load model, using fresh one. Error: {e}")
    
    game_history = []
    move_count = 0
    
    # Play until the game is over
    while game.is_terminal() is None:
        move_count += 1
        current_state = game.copy()
        
        # Run MCTS to get the best move and value estimate
        mcts = MCTS_Deep(current_state, model)
        est_value, visit_counts = mcts.search(num_simulations=num_simulations)
        
        if not visit_counts:
            break  # No valid moves found
        
        # Convert visit counts to a policy vector
        policy = _visits_to_policy(visit_counts)
        
        # Store the state, value, and policy for training
        if bootstrap:
            game_history.append((current_state.get_feature_plane(), est_value, policy))
        else:
            game_history.append((current_state.get_feature_plane(), policy))
        
        # Select the next action based on the visit counts
        action = _select_action(visit_counts, temperature=temperature)
        
        # Apply the move to the game state
        try:
            game = game.step(action)
        except Exception as e:
            print(f"Worker {game_id}: Error applying action {action}: {e}")
            break
    
    # Game finished, get the final outcome
    outcome = game.is_terminal()
    if outcome is None:
        outcome = 0 # If game ended prematurely, count as a draw
    
    # Process the game history to create training data
    result = []
    if bootstrap:
        for state_tensor, est_value, policy in game_history:
            result.append((state_tensor, est_value, policy))
    else:
        for state_tensor, policy in game_history:
            result.append((state_tensor, outcome, policy))
    
    return result

# Helper functions (moved outside the class for pickling)
def _visits_to_policy(visits):
    """Convert visit counts to a policy tensor for Connect4."""
    policy = torch.zeros(7, dtype=torch.float32)
    total_visits = sum(visits.values())
    
    if total_visits > 0:
        for col, count in visits.items():
            policy[col] = count / total_visits
    else:
        # Fallback to a uniform distribution if no visits were made
        policy.fill_(1.0 / 7.0)
            
    return policy

def _select_action(visits, temperature=1.0):
    """Select an action based on visit counts and temperature."""
    if not visits:
        raise ValueError("No valid actions provided")
        
    if temperature == 0:
        return max(visits.items(), key=lambda item: item[1])[0]
    
    actions = list(visits.keys())
    counts = np.array([visits[action] for action in actions])
    
    # Apply temperature and handle potential numerical issues
    if temperature != 1.0:
        counts = (counts + 1e-8) ** (1.0 / temperature)
        
    probs = counts / np.sum(counts)
    
    # Sample action, with a fallback to greedy selection
    try:
        probs = np.nan_to_num(probs, nan=1.0/len(actions))
        probs /= probs.sum()
        return np.random.choice(actions, p=probs)
    except Exception:
        return actions[np.argmax(counts)]

class Trainer:
    def __init__(self, game_class, model=None):
        """Initialize the trainer with a game class and model."""
        self.game_class = game_class
        
        if model is None:
            self.model = Connect4CNN()
        else:
            self.model = model
            
        os.makedirs("models", exist_ok=True)

    def get_data(self, num_games=10, temperature=1.0, num_simulations=300, bootstrap=False, parallel=False, num_workers=None):
        """Collect training data by self-play with MCTS."""
        if parallel and num_games > 1:
            return self.get_data_parallel(num_games, temperature, num_simulations, bootstrap, num_workers)
        else:
            # Fallback to sequential for a single game or if parallel is disabled
            return self.get_data_sequential(num_games, temperature, num_simulations, bootstrap)

    def get_data_parallel(self, total_games=10, temperature=1.0, num_simulations=300, bootstrap=False, num_workers=None):
        """Collect training data in parallel using a thread pool."""
        if num_workers is None:
            import multiprocessing
            num_workers = max(1, multiprocessing.cpu_count() - 1)
        num_workers = min(num_workers, total_games)
        
        # Save the current model to a temporary file for workers to load
        self.model.save("temp_model.pth")
        
        print(f"Starting {total_games} games with {num_workers} parallel workers")
        all_training_data = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(play_game, i, self.game_class, num_simulations, temperature, bootstrap) for i in range(total_games)]
            
            # Process results as they complete with a progress bar
            for future in tqdm(concurrent.futures.as_completed(futures), total=total_games, desc="Playing games"):
                try:
                    game_data = future.result()
                    all_training_data.extend(game_data)
                except Exception as e:
                    print(f"A game worker failed with an error: {e}")
        
        print(f"Completed {total_games} games, collected {len(all_training_data)} training examples")
        
        # Clean up the temporary model file
        try:
            os.remove("temp_model.pth")
        except OSError as e:
            print(f"Error removing temporary model file: {e}")
            
        return all_training_data

    def get_data_sequential(self, num_games=10, temperature=1.0, num_simulations=300, bootstrap=False):
        """Collect training data sequentially."""
        all_training_data = []
        for i in range(num_games):
            game_data = play_game(i, self.game_class, num_simulations, temperature, bootstrap)
            all_training_data.extend(game_data)
        return all_training_data

    def train(self, num_epochs=10, batch_size=32, lr=0.001, num_games=50, num_simulations=100, bootstrap=True, parallel=True, num_workers=None):
        """Train the model using self-play data."""
        temperature = 1.0
        for epoch in range(num_epochs):
            start_time = time.time()
            print(f"Epoch {epoch+1}/{num_epochs}")
            
            # Collect training data
            training_data = self.get_data(
                num_games=num_games, 
                temperature=temperature, 
                num_simulations=num_simulations,
                bootstrap=bootstrap, 
                parallel=parallel,
                num_workers=num_workers
            )
            temperature = max(temperature * 0.95, 0.1)  # Slower decay

            if not training_data:
                print("No training data collected, skipping epoch.")
                continue
                
            # Prepare data for training
            state_tensors = torch.stack([torch.FloatTensor(s) for s, v, p in training_data])
            value_targets = torch.tensor([v for s, v, p in training_data], dtype=torch.float32).view(-1, 1)
            policy_targets = torch.stack([p for s, v, p in training_data])
            
            dataset = torch.utils.data.TensorDataset(state_tensors, value_targets, policy_targets)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # Set up optimizer and train
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
            self.model.train()
            total_loss = 0
            
            for states, values, policies in dataloader:
                optimizer.zero_grad()
                
                pred_values, pred_policies = self.model._network_forward(states.unsqueeze(1))
                
                value_loss = F.mse_loss(pred_values, values)
                policy_loss = F.cross_entropy(pred_policies, policies)
                
                loss = value_loss + policy_loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader) if dataloader else 0
            epoch_time = time.time() - start_time
            
            print(f"Epoch {epoch+1} completed. Avg loss: {avg_loss:.4f} Time: {epoch_time:.1f}s")
            self.save_model(f"models/connect4_model_epoch_{epoch+1}.pth")
    
    def save_model(self, path):
        self.model.save(path)
    
    def load_model(self, path):
        self.model.load(path)
        return self.model

if __name__ == "__main__":
    trainer = Trainer(C4)
    
    trainer.train(
        num_epochs=10, 
        batch_size=64, 
        num_games=50, 
        num_simulations=100,
        bootstrap=True, 
        parallel=True,
        num_workers=4
    )
    
    trainer.save_model("models/connect4_final_model.pth")