"""
This module implements a Monte Carlo Tree Search (MCTS) algorithm using a neural network model for state evaluation and action selection.
This MCTS algorithm is designed to work with a game environment that provides methods for state management, action generation, and reward evaluation.
It has been designed to be flexible and support various game types, including adversarial games. 
If one wishes to use this MCTS with a specific game, they only need to ensure that the game state class has the necessary methods implemented,
and the neural network model is compatible with the state representation used in the game. 

Requirements from the game state class:
1. `copy()`: Method to create a copy of the current game.
2. `is_terminal()`: Method to check if the game state is terminal and return the outcome if it is (None if not terminal)
3. `step(action)`: Method to apply an action to the current state and return the resulting deep copy of the new game.
4. `valid_moves()`: Method to return a list of valid actions that can be taken from the current state. (Recommend using tuples)
5. `get_feature_plane()`: Method to return the feature plane representation of the game state for the neural network model.
6. `player`: Property to get the current player of the game state (1 for player, -1 for opponent).

Requirements from the neural network model:
1. `forward(state)`: Method to take a game state and return the value and policy (action probabilities).
    It returns a tuple where the first element is the value estimation and the second element is a dictionary of action probabilities. 
2. `get_value(state)`: Method to get the value estimation for a given state.

Notes:
- The MCTS algorithm uses a tree structure where each node represents a game state.
- The nodes are indexed by actions, and each node keeps track of its children, visit counts, value estimates, and prior probabilities.
- Actions are represented as keys in dictionaries, allowing for flexible action representation (e.g., tuples for coordinates).
- I represented actions as coordinates in the policy head for ease of use with grid-based games, but this can be adapted for other action representations.
"""
import numpy as np


class Deep_Node:
    """
    This class represents a node in the Monte Carlo Tree Search (MCTS) tree.
    This node is only designed for complete information and deterministic games.
    It uses a neural network model to evaluate the state and provide action probabilities.
    Each node contains:
    - `state`: The game state represented by this node.
    - `model`: The neural network model used for state evaluation and action probabilities.
    - `children`: A dictionary mapping actions to child nodes.
    - `Q`: A dictionary mapping actions to Q-values (expected rewards).
    - `N`: A dictionary mapping actions to visit counts.
    - `P`: A dictionary mapping actions to policy probabilities (from the model).
    - `T`: A dictionary mapping actions to the player of the next state (1 for current player, -1 for opponent).

    Search works by selecting actions based on the PUCT (Policy Upper Confidence Tree) formula,
    expanding nodes when necessary, and backing up values through the tree.

    What makes this node "deep" is that it uses a neural network model to evaluate the state and provide action probabilities,
    allowing for more informed decision-making compared to traditional MCTS nodes that rely solely on random simulations.
    """
    def __init__(self, state, model):
        self.state = state.copy()       # Deep copy of the game state to ensure immutability
        self.model = model              # Neural network model for state evaluation and action probabilities
        self.children = {}              # Children nodes indexed by action
        self.Q        = {}              # Q-values for actions (with respect to the next player to move)
        self.N        = {}              # Number of visits for actions
        self.P        = {}              # Policy probabilities for actions (from the model)
        self.T        = {}              # Player of next state (1 if player stays the same, -1 if it changes)
        self.player   = state.player    # Current player of the state
        self.N_visits = 0               # Total number of visits to this node
        self.value    = 0               # Value of the node (with respect to the current player)
        self.terminal = False           # Whether the node is terminal (game over)
        self._initialize_node(state)

    def _initialize_node(self, state):
        outcome = state.is_terminal()   # Check if the state is terminal and get the outcome
        if outcome is not None:         # If terminal, set value and mark as terminal
            self.terminal = True
            self.value = outcome * self.player  # Set the value based on the outcome and current player
            return
        
        # Get value and policy from model
        self.value, self.P = self.model.forward(state)  # If not terminal, get value and policy from the model
        
        # Initialize tracking dictionaries for all valid actions
        for action in self.P:
            self.children[action] = None
            self.N[action] = 0
            
            # Create next state for Q-value initialization
            next_state = self.state.step(action)
            # self.Q[action] = self.model.get_value(next_state)
            self.Q[action] = 0.0 # Initialize Q-value to 0.0 for all actions (can be updated later) (this is done as a speed up)
            
            # T=1 for moves that lead to the current player, T=-1 for opponent's turn
            self.T[action] = 1 if (next_state.player == self.player) else -1


    def _select_action(self, puct=1.0):
        """Select action using PUCT formula"""
        if self.terminal or not self.P:
            return None
            
        # Calculate UCT values for all actions
        U = {}
        for action in self.P:
            if self.N[action] == 0:
                # If action has never been visited, set U to the prior probability
                U[action] = self.P[action] 
            
            exploration = puct * self.P[action] * np.sqrt(self.N_visits) / (1 + self.N[action])
            exploitation = self.Q[action]
            U[action] = exploitation * self.T[action] + exploration
            
        # Return action with maximum UCT value
        return max(U, key=U.get) if U else None
    

class MCTS_Deep:
    """
    This class implements the Monte Carlo Tree Search (MCTS) algorithm using a neural network model for state evaluation.
    It is designed to work with a game environment that provides the necessary methods for state management, action generation, and reward evaluation.
    The MCTS_Deep class uses the Deep_Node class to represent nodes in the search tree.
    It performs the search by selecting nodes based on the PUCT (Policy Upper Confidence Tree) formula,
    expanding nodes when necessary, and backing up values through the tree.

    Args:
        state: The initial game state to start the search from.
        model: The neural network model used for state evaluation and action probabilities.
        root: The root node of the search tree, initialized with the initial state and model.
    """
    def __init__(self, state, model):
        self.model = model
        self.state = state
        self.root = Deep_Node(state, model)

    def search(self, puct=1.0, num_simulations=1000):
        for _ in range(num_simulations):
            path = self._select(puct)
            node = self._expand(path)
            value = node.value
            self._backup(path, value)
        return self.root.value, self.root.N        # Returns the estimated value and visit counts for the children of root node

    def _select(self, puct=1.0):
        """Select a node to expand using PUCT."""
        path = []
        node = self.root
        
        # Get the best action from current node
        action = node._select_action(puct)
        
        # If no valid action, return the current path
        if action is None:
            return path
            
        path.append((node, action))
        
        # Continue selecting until we find a node that hasn't been initialized
        while action is not None and node.children[action] is not None:
            node = node.children[action]
            
            # If terminal node, stop selection
            if node.terminal:
                break
                
            # Get next action
            action = node._select_action(puct)
            if action is None:
                break
                
            path.append((node, action))
            
        return path

    def _expand(self, path):
        """Expand a node in the tree."""
        # If path is empty, return root
        if not path:
            return self.root
            
        node, action = path[-1]
        
        # If already expanded, return existing child (only happens with terminal nodes)
        if node.children[action] is not None:
            return node.children[action]
            
        # Otherwise create a new node
        next_state = node.state.step(action)  # Apply the action to get the next state
        node.children[action] = Deep_Node(next_state, self.model) # Create a new child node with the next state and model (handles initialization)
        
        return node.children[action]

    def _backup(self, path, value):
        """
        Backpropagate the value through the tree.
        
        Args:
            path: List of (node, action) pairs from root to leaf
            value: Value to backpropagate (from the perspective of the player at the leaf)
        """
        # If path is empty, nothing to do
        if not path:
            return
            
        # Start with the value from the leaf node's perspective (for this implementation, we assume the value of each node is with respect to itself)
        action = path[-1][1]
        leaf_player = path[-1][0].children[action].player
        
        for node, action in reversed(path):
            # Skip nodes without valid actions
            if action is None:
                continue
                
            # Update action visit counts
            node.N[action] += 1
            node.N_visits += 1
            
            # Update Q-value if child exists
            if node.children[action] is not None:
                child = node.children[action]
                node.Q[action] = child.value 
            
                
            # Update node value (weighted average)
            prev_value = node.value
            # Update value based on the leaf player's perspective (this is the value we are propagating up) (simple average)
            node.value = ((node.N_visits - 1) * prev_value + (value * (leaf_player * node.player))) / node.N_visits



