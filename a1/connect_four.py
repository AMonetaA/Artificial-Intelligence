from typing import List
from abc import ABC, abstractmethod
from copy import deepcopy
from random import randint


class State:
    """
    Represents the Connect Four board.

    """

    def __init__(self, num_cols: int, num_rows: int, turn: str, board: list[list[str]] = None):
        """
        Stores the board's size and the grid of pieces in play.

        :param num_cols: The number of columns (width) of the board
        :type num_cols: int
        :param num_rows: The number of rows (height) of the board
        :type num_rows: int
        :param  turn: Whose turn is next
        :type turn: str (one of 'x' or 'o')
        :param board: The current state of pieces in the grid. Note: coordinates are grid[column][row]
        :            where grid[0][0] is the bottom left corner.
        :            If none is given, then an empty grid will be generated.
        :type board: List[List[str]], where strings should be one of '.' (empty position), 'x', or 'o'
        """

        self.num_rows = num_rows
        self.num_cols = num_cols
        self.turn = turn
        self.board = board

        if self.board is None:
            self.board = []
            for i in range(num_cols):
                new_row = []
                for j in range(num_rows):
                    new_row.append('.')
                self.board.append(new_row)
        elif num_cols <= 0 or num_rows <= 0 or len(self.board) != num_cols or len(self.board[0]) != num_rows:
            print("Warning: Board was initialized with incorrect size.")

        self.is_terminal = False
        self.winner = self.four_in_a_row()
        self.is_terminal = self.winner != '' or len(self.get_legal_moves()) == 0


    def __eq__(self, other):
        """
        Compares two States, producing true if they represent the same board, and false otherwise

        :param  other: The other State
        :type other: State
        :return: True if equal, False otherwise
        :rtype: bool
        """
        return self.board == other.board


    def coords_legal(self, col: int, row: int):
        """
        Produces True if the given row and column are in-bounds for this board (indexed from 0)

        :param col: The column number
        :type col: int
        :param row: The row number
        :type row: int
        :return: True if in-bounds, false otherwise
        :rtype: bool
        """
        return col >= 0 and col < self.num_cols and row >= 0 and row < self.num_rows


    def four_in_a_row(self):
        """
        If the current board contains a vertical, horizontal, or diagonal four-in-a-row of 'x' or 'o',
        produce the piece that has the four-in-a-row

        :return: '' if there are no four-in-a-rows, 'x' if there are four 'x's in a row, and 'o' if there
        :        are four 'o's in a row
        :rtype: str
        """

        result = ''
        found = False
        col_dirs = [1, 1, 0, -1]
        row_dirs = [0, 1, 1, 1]

        i = 0
        while i < self.num_cols and not found:
            j = 0
            while j < self.num_rows and not found:
                if self.board[i][j] != '.':
                    piece = self.board[i][j]
                    dir = 0
                    while dir < len(col_dirs) and not found:
                        farthest_pnt = [i + 3 * col_dirs[dir], j + 3 * row_dirs[dir]]
                        if self.coords_legal(farthest_pnt[0], farthest_pnt[1]):
                            if piece == self.board[i + col_dirs[dir]][j + row_dirs[dir]] and \
                                    piece == self.board[i + 2 * col_dirs[dir]][j + 2 * row_dirs[dir]] and \
                                    piece == self.board[farthest_pnt[0]][farthest_pnt[1]]:
                                result = piece
                                found = True
                        dir += 1
                j += 1
            i += 1
        return result


    def move_is_legal(self, col: int):
        """
        Produces True if it is legal to drop a piece into the given column, and False otherwise

        :param col: The column number
        :type col: int
        :return: True if legal, False otherwise
        :rtype: bool
        """
        return col >= 0 and col < self.num_cols and self.board[col][self.num_rows-1] == '.' and not self.is_terminal


    def get_legal_moves(self):
        """
        Produces a list of legal moves (i.e. columns indices that a piece may be dropped into),
        given the current state

        :return: A list of legal moves
        :rtype: List[int]
        """
        result = []
        if not self.is_terminal:
            for i in range(self.num_cols):
                if self.move_is_legal(i):
                    result.append(i)
        return result


    def get_first_empty_row(self, col: int):
        """
        Produces the index of the first empty row in the given column, or -1 if there is no such row

        :param col: The column number
        :type col: int
        :return: The index of the first empty row, or -1 if none exists
        :rtype: int
        """
        result = -1
        if col >= 0 and col < self.num_cols:
            for i in range(self.num_rows):
                if self.board[col][i] == '.':
                    result = i
                    break
        return result


    def peek_next_board(self, move: int):
        """
        Produces the board that results from playing the given move. The piece placed ('x' or 'o') corresponds
        to the current turn. Note: this function does not alter the current board.

        :param move: The column index to insert a piece into
        :type move: int
        :return: The board resulting from the given move being played, or None if the move is illegal
        :rtype: List[List[str]]
        """
        result = None
        if self.move_is_legal(move):
            result = deepcopy(self.board)
            place_row = self.get_first_empty_row(move)
            result[move][place_row] = self.turn
        return result


    def advance_state(self, move: int):
        """
        Updates the state by playing the given move. A piece is added to the board, and the turn alternates
        from one player to the other

        :param move: The column index to insert a piece into
        :type move: int
        :return: True if successful, False if the given move was illegal
        :rtype: bool
        """
        result = False
        if not self.is_terminal and self.move_is_legal(move):
            place_row = self.get_first_empty_row(move)
            self.board[move][place_row] = self.turn
            if self.turn == 'x':
                self.turn = 'o'
            else:
                self.turn = 'x'
            self.winner = self.four_in_a_row()
            self.is_terminal = self.winner != '' or len(self.get_legal_moves()) == 0
            result = True
        return result


    def display(self):
        """
        Prints out the board in a human-readable format.

        For example:
        +---------------+
        | . . . . . . . |
        | . . . . . . . |
        | . . x x . . . |
        | . . o o . . . |
        | . . o x . . . |
        | . x x o . . o |
        +---------------+
        """

        def print_cap():
            print('+', end='')
            for i in range(self.num_cols * 2 + 1):
                print('-', end='')
            print('+')

        def print_grid_line(row_index):
            print_str = '| '
            for i in range(self.num_cols):
                print_str += self.board[i][row_index] + ' '
            print_str += '|'
            print(print_str)

        print_cap()
        for i in range(self.num_rows):
            print_grid_line(self.num_rows - 1 - i)
        print_cap()



class Player(ABC):
    """
    This abstract class provides for an interface for Connect Four playing agents.
    All agents must override the abstract methods below.

    """

    @abstractmethod
    def initialize(self, role: str):
        """
        This function is called once for each agent at the beginning of a game, before any moves are made

        :param role: The role of the player
        :type role: str (one of 'x' or 'o')
        """
        pass

    @abstractmethod
    def play(self, state: State):
        """
        This function is called every time it is the player's turn. It produces the column number that a
        piece should be dropped into

        :param state: the game's current State
        :type state: State
        :return: A column number representing a valid move to be played
        :rtype: int
        """
        pass



class Game:
    """
    Represents a game of Connect Four between two players

    """
    def __init__(self, p1: Player, p2: Player, state: State = None):
        """
        Stores player and game state information

        :param p1: Player 1
        :type p1: Player
        :param p2: Player 2
        :type p2: Player
        :param state: The initial state
        :type state: State
        """
        self.__p1 = p1
        self.__p2 = p2
        if state is None:
            self.__game_state = State(7,6,'x')
        else:
            self.__game_state = state


    def play_game(self):
        """
        Carries out a game between the two players

        :return: The winner of the game
        :rtype: str ('x' if x won, 'o' if o won, or '' if a draw)
        """
        self.__p1.initialize('x')
        self.__p2.initialize('o')

        while(not self.__game_state.is_terminal):
            if(self.__game_state.turn == 'x'):
                move = self.__p1.play(deepcopy(self.__game_state))
            else:
                move = self.__p2.play(deepcopy(self.__game_state))
            self.__game_state.advance_state(move)
        return self.__game_state.winner


    def display(self):
        """
        Prints a representation of the game to the terminal. If the game is over, print a message
        declaring the winner.

        """
        self.__game_state.display()
        if self.__game_state.is_terminal:
            if self.__game_state.winner == '':
                print("It's a draw!")
            else:
                print("The winner is: " + self.__game_state.winner + "!")
