import numpy as np

class C4:
    def __init__(self):
        # Standard Connect4 board is 6 rows x 7 columns
        # Note: We store as [column][row] for easier column-based operations
        self.board = np.zeros((7, 6), dtype=int)
        self.current_player = 1
        self.player = 1  # Alias for compatibility with MCTS
        self.moves_made = 0

    def reset(self):
        return C4()
    
    def get_feature_plane(self):
        """
        Get the feature plane for the current board state.
        
        Returns:
            A numpy array of shape (6, 7) representing the board.
            The current player's pieces are 1, opponent's are -1.
        """
        # Convert to row-major format for model (6x7)
        return np.transpose(self.board) * self.current_player
    
    def valid_moves(self):
        """
        Get the valid moves for the current player.
        
        Returns:
            A list of valid column indices where a piece can be dropped.
        """
        return [col for col in range(7) if 0 in self.board[col]]
    
    def drop_piece(self, column):
        """
        Drop a piece into the specified column for the current player.
        
        Args:
            column: The column index (0-6) where the piece should be dropped.
        
        Returns:
            True if the move was successful, False if the column is full.
        """
        # Find the lowest empty row in the column
        for row in range(5, -1, -1):
            if self.board[column][row] == 0:
                self.board[column][row] = self.current_player
                self.moves_made += 1
                return True
        return False
    
    def switch_player(self):
        """
        Switch the current player.
        """
        self.current_player = -self.current_player
        self.player = self.current_player

    def step(self, action):
        """
        Perform a step in the game by dropping a piece in the specified column.
        
        Args:
            action: The column index (0-6) where the piece should be dropped.
        
        Returns:
            A new game state after the action is applied.
        """
        new_state = self.copy()
        success = new_state.drop_piece(action)
        if not success:
            raise ValueError(f"Invalid move: Column {action} is full.")
        new_state.switch_player()
        return new_state
    
    def copy(self):
        """
        Create a copy of the current game state.
        
        Returns:
            A new C4 instance with the same board and current player.
        """
        new_game = C4()
        new_game.board = np.copy(self.board)
        new_game.current_player = self.current_player
        new_game.player = self.current_player
        new_game.moves_made = self.moves_made
        return new_game
    
    def is_terminal(self):
        """
        Check if the game is in a terminal state (win or draw).
        
        Returns:
            1 if player 1 won, -1 if player 2 won, 0 if draw, None if game is ongoing.
        """
        # The issue is in the win detection logic:
        # We're storing the board as [col][row] but our win checks are assuming wrong access pattern
        
        # Check for horizontal win (need to fix access pattern)
        for row in range(6):
            for col in range(4):
                if (self.board[col][row] != 0 and 
                    self.board[col][row] == self.board[col+1][row] == 
                    self.board[col+2][row] == self.board[col+3][row]):
                    # This player has won
                    return self.board[col][row]
        
        # Check for vertical win
        for col in range(7):
            for row in range(3):
                if (self.board[col][row] != 0 and 
                    self.board[col][row] == self.board[col][row+1] == 
                    self.board[col][row+2] == self.board[col][row+3]):
                    # This player has won
                    return self.board[col][row]
        
        # Check for diagonal win (bottom-left to top-right)
        for col in range(4):
            for row in range(3, 6):
                if (self.board[col][row] != 0 and 
                    self.board[col][row] == self.board[col+1][row-1] == 
                    self.board[col+2][row-2] == self.board[col+3][row-3]):
                    # This player has won
                    return self.board[col][row]
        
        # Check for diagonal win (top-left to bottom-right)
        for col in range(4):
            for row in range(3):
                if (self.board[col][row] != 0 and 
                    self.board[col][row] == self.board[col+1][row+1] == 
                    self.board[col+2][row+2] == self.board[col+3][row+3]):
                    # This player has won
                    return self.board[col][row]
        
        # Check for draw - ONLY if the board is actually full
        if self.moves_made >= 42:  # 6x7 board is full
            return 0
        
        # Game is not over
        return None
        
    def __str__(self):
        """Return a string representation of the board for debugging."""
        board_str = ""
        for row in range(6):
            row_str = "|"
            for col in range(7):
                if self.board[col][row] == 0:
                    row_str += " "
                elif self.board[col][row] == 1:
                    row_str += "X"
                else:
                    row_str += "O"
                row_str += "|"
            board_str = row_str + "\n" + board_str
        return board_str + "-" * 15