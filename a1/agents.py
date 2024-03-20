from copy import deepcopy
from heapq import heappush, heappop
import time
import argparse
from typing import List
from connect_four import State, Game, Player
from copy import deepcopy
from random import randint

class MinimaxNode:
    """
    One node in the Minimax search tree.

    """
    def __init__(self, state: State):
        """
        Stores the node's state, (heuristic) value, and a dictionary of successor nodes,
        where the keys are possible moves and the values are the successor nodes resulting from those moves

        :param  state: The state associated with this node
        :type state: State
        """
        self.state = state
        self.value = 0
        self.successors = {}

    def __eq__(self, other):
        """
        Recursively compares two MinimaxNodes, producing true if all have the same heuristic value,
        state, and successors. Used by the == operator.

        :param  other: The other MinimaxNode
        :type other: MinimaxNode
        :return: True if the nodes are equal, False otherwise
        :rtype: bool
        """
        return self.state == other.state and self.value == other.value and self.successors == other.successors


def createChild(node: MinimaxNode):
    moves = node.state.get_legal_moves()
    for i in moves:
        # state of the child, choosing ith column, alter the board and turn
        nextState = deepcopy(node.state)
        nextState.board = deepcopy(node.state.peek_next_board(i))
        nextState.turn = 'o' if node.state.turn == 'x' else 'x'
        # append the successors
        nextNode = MinimaxNode(nextState)
        node.successors[i] = nextNode

    return 0

def minimax(node: MinimaxNode, depth: int, max_role: str, heuristic_fn):
    """
    Performs minimax search from the given node out to a maximum depth, when heuristic evaluation is performed.
    Generates a tree of MinimaxNodes rooted at node, with correct state, value, and successors attributes

    :param node: The node that will be the root of this search
    :type node: MinimaxNode
    :param depth: The search depth. When depth is 0, perform a heuristic evaluation.
    :type depth: int
    :param max_role: The maximizing player
    :type max_role: str (one of 'x' or 'o')
    :param heuristic_fn: The heuristic evaluation function to be used at the max search depth
    :type heuristic_fn: Function (State str -> float), which consumes the state to be evaluated and
    :                   the maximizing player's role (either 'x' or 'o')
    :return: The evaluation of the given node
    :rtype: int
    """
    # Create childrene for the current node
    #createChild(node)

    if depth == 0 or node.state.is_terminal:
        node.value = heuristic_fn(node.state,max_role)
        return node.value
    
    moves = node.state.get_legal_moves()
    for i in moves:
        # state of the child, choosing ith column, alter the board and turn
        nextState = deepcopy(node.state)
        nextState.advance_state(i)
        # append the successors
        nextNode = MinimaxNode(nextState)
        node.successors[i] = nextNode
        minimax(nextNode, depth-1, max_role, heuristic_fn)
    
    if max_role == node.state.turn:
        node.value = float('-inf')
        for child in node.successors.values():
            node.value = max(node.value, child.value)
    else:
        node.value = float('inf')
        for child in node.successors.values():
            node.value = min(node.value, child.value)

    return node.value



def three_line_heur(state: State, max_role: str):
    """
    Performs a heuristic evaluation of the given state, equal to the number of three-in-a-rows for the
    maximizing player minus the number of three-in-a-rows for the minimizing player.
    If the state is terminal, gives the true evaluation instead (100 if the maximizer has won,
    0 for a draw, or -100 if the minimizer has won)

    :param state: The state to evaluate
    :type state: State
    :param max_role: The role of the maximizing player
    :type max_role: str (one of 'x' or 'o')
    :return: The evaluation of the given state
    :rtype: int
    """
    col_dirs = [1, 1, 0, -1]
    row_dirs = [0, 1, 1, 1]
    result = 0

    #If the state is terminal, give the true evaluation
    if state.is_terminal:
        if state.winner == '':
            return 0
        elif state.winner == max_role:
            return 100
        else:
            return -100

    #If the state is not terminal, give the heuristic evaluation
    i = 0
    while i < state.num_cols:
        j = 0
        while j < state.num_rows:
            if state.board[i][j] != '.':
                piece = state.board[i][j]
                dir = 0
                while dir < len(col_dirs):
                    farthest_pnt = [i + 2 * col_dirs[dir], j + 2 * row_dirs[dir]]
                    if state.coords_legal(farthest_pnt[0], farthest_pnt[1]):
                        if piece == state.board[i + col_dirs[dir]][j + row_dirs[dir]] and \
                                piece == state.board[farthest_pnt[0]][farthest_pnt[1]]:
                            if piece == max_role:
                                result += 1
                            else:
                                result -= 1
                    dir += 1
            j += 1
        i += 1
    return result


def zero_heur(state: State, max_role: str):
    """
    Produces 0 for any non-terminal state.
    If the state is terminal, gives the true evaluation instead (100 if the maximizer has won,
    0 for a draw, or -100 if the minimizer has won)

    :param state: The state to evaluate
    :type state: State
    :param max_role: The role of the maximizing player
    :type max_role: str (one of 'x' or 'o')
    :return: The evaluation of the given state
    :rtype: int
    """

    #If the state is terminal, give the true evaluation
    if state.is_terminal:
        if state.winner == '':
            return 0
        elif state.winner == max_role:
            return 100
        else:
            return -100

    #If the state is not terminal, produce 0
    return 0



def my_heuristic(state: State, max_role: str):
    """
    Performs a heuristic evaluation of the given state.
    If the state is terminal, gives the true evaluation instead (100 if the maximizer has won,
    0 for a draw, or -100 if the minimizer has won)

    :param state: The state to evaluate
    :type state: State
    :param max_role: The role of the maximizing player
    :type max_role: str (one of 'x' or 'o')
    :return: The evaluation of the given state
    :rtype: int
    """
    # Your code here!
    col_dirs = [1, 1, 0, -1]
    row_dirs = [0, 1, 1, 1]
    result = 0
    result2 = 0

    if state.is_terminal:
        if state.winner == '':
            return 0
        elif state.winner == max_role:
            return 100
        else:
            return -100
        
    #If the state is not terminal, give the heuristic evaluation
    i = 0
    while i < state.num_cols:
        j = 0
        while j < state.num_rows:
            if state.board[i][j] != '.':
                piece = state.board[i][j]
                dir = 0
                while dir < len(col_dirs):
                    
                    farthest_pnt = [i + 2 * col_dirs[dir], j + 2 * row_dirs[dir]]
                    if state.coords_legal(farthest_pnt[0], farthest_pnt[1]):
                        if piece == state.board[i + col_dirs[dir]][j + row_dirs[dir]] and \
                                piece == state.board[farthest_pnt[0]][farthest_pnt[1]]:
                            
                            next_pnt = [i + 3 * col_dirs[dir], j + 3 * row_dirs[dir]]
                            if state.coords_legal(next_pnt[0], next_pnt[1]) and \
                                '.' == state.board[next_pnt[0]][next_pnt[1]]:
                                if piece == max_role:
                                    result += 1
                                else:
                                    result -= 1
                            
                            pre_pnt = [i - col_dirs[dir], j - row_dirs[dir]]
                            if state.coords_legal(pre_pnt[0], pre_pnt[1]) and \
                                '.' == state.board[pre_pnt[0]][pre_pnt[1]]:
                                if piece == max_role:
                                    result += 1
                                else:
                                    result -= 1

                        elif piece == state.board[i + col_dirs[dir]][j + row_dirs[dir]] and \
                                '.' == state.board[farthest_pnt[0]][farthest_pnt[1]]:
                            if piece == max_role:
                                result2 += 1
                            else:
                                result2 -= 1
                            
                            pre_pnt = [i - col_dirs[dir], j - row_dirs[dir]]
                            if state.coords_legal(pre_pnt[0], pre_pnt[1]) and \
                                '.' == state.board[pre_pnt[0]][pre_pnt[1]]:
                                if piece == max_role:
                                    result += 1
                                else:
                                    result -= 1

                    dir += 1
            j += 1
        i += 1
    return result*2 + result2 * (1/10)



class MinimaxPlayer(Player):
    """
    An agent that uses minimax to select moves.

    """

    def __init__(self, depth: int, heur, display=True):
        """
        Stores minimax parameters

        :param depth: The depth at which search is terminated and a heuristic evaluation is performed
        :type depth: int
        :param heur: The heuristic evaluation function to be used at the max search depth
        :type heur: Function (State str -> float), which consumes the state to be evaluated and
        :           the maximizing player's role (either 'x' or 'o')
        :param display: If true, print board every play
        :type display: bool
        """
        self.role = ''
        self.depth = depth
        self.heur = heur
        self.display = display

    def initialize(self, role: str):
        """
        This function is called once for each agent at the beginning of a game, before any moves are made

        :param role: The role of the player
        :type role: str (one of 'x' or 'o')
        """
        self.role = role

    def play(self, state: State):
        """
        This function is called every time it is the player's turn. It produces the column number that a
        piece should be dropped into

        :param state: the game's current State
        :type state: State
        :return: A column number representing a valid move to be played
        :rtype: int
        """
        if self.display:
            state.display()
        root = MinimaxNode(state)
        minimax(root, self.depth, self.role, self.heur)
        best_moves = []
        for move in root.successors.keys():
            if len(best_moves) == 0 or root.successors[move].value > root.successors[best_moves[0]].value:
                best_moves = [move]
            elif root.successors[move].value == root.successors[best_moves[0]].value:
                best_moves.append(move)
        return best_moves[randint(0, len(best_moves)-1)]



class FirstMovePlayer(Player):
    """
    An agent that always plays the first legal move.

    """

    def initialize(self, role: str):
        pass

    def play(self, state: State):
        state.display()
        return state.get_legal_moves()[0]



class RandomPlayer(Player):
    """
    An agent that always plays a random move.

    """

    def initialize(self, role: str):
        pass

    def play(self, state: State):
        state.display()
        moves = state.get_legal_moves()
        return moves[randint(0,len(moves)-1)]


class HumanPlayer(Player):
    """
    An agent that allows you to play by keyboard input!

    """

    def initialize(self, role: str):
        pass

    def play(self, state: State):
        state.display()
        moves = state.get_legal_moves()
        valid = False
        col = -1
        while not valid:
            str = input("Enter a column number in the range [0,6]: ")
            try:
                col = int(str)
                if col in moves:
                    valid = True
                else:
                    print("Selected column is not valid.")
            except ValueError:
                print("Unable to parse input.")
        return col


def run20 (p1:Player, p2:Player):
    i=0
    win = 0
    draw = 0
    lose = 0
    while i < 20:
        game = Game(p1, p2)
        winner = game.play_game()
        if winner == 'x': win += 1
        elif winner == 'o': lose += 1
        else: draw += 1
        i+=1
    print(f"win: {win} draw: {draw} lose: {lose}")
    return 0

if __name__ == "__main__":

    # start = time.time()
    # # This is the code that gets run when you run this file. You can set up games to be played here.

    # game = Game(HumanPlayer(), MinimaxPlayer(4, my_heuristic))
    # # Here are some more examples of game initialization:
    # # game = Game(MinimaxPlayer(4, three_line_heur), MinimaxPlayer(4, zero_heur))
    # # game = Game(RandomPlayer(), FirstMovePlayer())
    # winner = game.play_game()
    # game.display()
    # end = time.time()

    # print(end - start)
    run20(MinimaxPlayer(4, my_heuristic),MinimaxPlayer(4, three_line_heur))