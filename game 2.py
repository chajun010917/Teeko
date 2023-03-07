import copy
import random
import numpy as np
import math

class TeekoPlayer:
    """ An object representation for an AI game player for the game Teeko.
    """
    board = [[' ' for j in range(5)] for i in range(5)]
    pieces = ['b', 'r']
    perfect_weight = [[0, 1, 1, 1, 0],
                      [1, 2, 2, 2, 1],
                      [1, 2, 3, 2, 1],
                      [1, 2, 2, 2, 1],
                      [0, 1, 1, 1, 0]]

    def __init__(self):
        """ Initializes a TeekoPlayer object by randomly selecting red or black as its
        piece color.
        """
        self.my_piece = random.choice(self.pieces)
        self.opp = self.pieces[0] if self.my_piece == self.pieces[1] else self.pieces[1]

    def make_move(self, state):
        """ Selects a (row, col) space for the next move. You may assume that whenever
        this function is called, it is this player's turn to move.

        Args:
            state (list of lists): should be the current state of the game as saved in
                this TeekoPlayer object. Note that this is NOT assumed to be a copy of
                the game state and should NOT be modified within this method (use
                place_piece() instead). Any modifications (e.g. to generate successors)
                should be done on a deep copy of the state.

                In the "drop phase", the state will contain less than 8 elements which
                are not ' ' (a single space character).

        Return:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

        Note that without drop phase behavior, the AI will just keep placing new markers
            and will eventually take over the board. This is not a valid strategy and
            will earn you no points.
        """
        drop_phase = (sum(a.count(self.my_piece) for a in self.board) < 4)   # TODO: detect drop phase

        if not drop_phase: #gotta make the best move
            # TODO: choose a piece to move and remove it from the board
            # (You may move this condition anywhere, just be sure to handle it)
            #
            # Until this part is implemented and the move list is updated
            # accordingly, the AI will not follow the rules after the drop phase!
            moves = self.succ(state)
            a, ind = 0, 0
            for move in moves:
                tempState = copy.deepcopy(state)
                tempState[move[1][0]][move[1][1]] = ' '
                tempState[move[0][0]][move[0][1]] = self.my_piece
                temp = self.max_value(tempState, 2)
                if temp>a:
                    a=temp
                    ind = moves.index(move)
            move = list(moves[ind])

        else: #when all four markers are not on the board, try to take the best spot
            moves = self.succ(state)
            a, ind, tempVal = 0, 0, 0
            for move in moves:
                tempState = copy.deepcopy(state)
                tempState[move[0]][move[1]] = self.my_piece
                d, g = self.combined_distance(tempState)
                dis = self.perfect_weight[move[0]][move[1]]/d
                if dis>tempVal:
                    tempVal = dis
                    ind = moves.index(move)
                if self.game_value(tempState) == 1:
                    ind = moves.index(move)
                    break
            move = [tuple(moves[ind])]
        return move

    def succ (self, state): # check which moves are valid
        if self.game_value(state) != 0:
            return self.game_value(state)
        dir = [[0, +1], [+1, -1], [+1, 0], [+1, +1], [-1, -1], [-1, 0], [-1, +1], [0, -1]]
        currLoc, valid1 , valid2 = [], [], []
        for x in range(5):
            for y in range(5):
                if state[x][y] == self.my_piece:
                    currLoc.append((x,y))
                elif state[x][y] == ' ':
                    valid1.append((x,y))
        if len(currLoc) < 4: #drop phase
            return valid1
        else:
            for x in currLoc:
                for y in dir:
                    newX = x[0] + y[0]
                    newY = x[1] + y[1]
                    if ( 0 <= newX <= 4 and 0 <= newY <= 4) and state[newX][newY] == ' ':
                        valid2.append([(newX, newY), (x[0],x[1])])
            return valid2

    def opponent_move(self, move):
        """ Validates the opponent's next move against the internal board representation.
        You don't need to touch this code.

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.
        """
        # validate input
        if len(move) > 1:
            source_row = move[1][0]
            source_col = move[1][1]
            if source_row != None and self.board[source_row][source_col] != self.opp:
                self.print_board()
                print(move)
                raise Exception("You don't have a piece there!")
            if abs(source_row - move[0][0]) > 1 or abs(source_col - move[0][1]) > 1:
                self.print_board()
                print(move)
                raise Exception('Illegal move: Can only move to an adjacent space')
        if self.board[move[0][0]][move[0][1]] != ' ':
            raise Exception("Illegal move detected")
        # make move
        self.place_piece(move, self.opp)

    def place_piece(self, move, piece):
        """ Modifies the board representation using the specified move and piece

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

                This argument is assumed to have been validated before this method
                is called.
            piece (str): the piece ('b' or 'r') to place on the board
        """
        if len(move) > 1:
            self.board[move[1][0]][move[1][1]] = ' '
        self.board[move[0][0]][move[0][1]] = piece

    def print_board(self):
        """ Formatted printing for the board """
        for row in range(len(self.board)):
            line = str(row)+": "
            for cell in self.board[row]:
                line += cell + " "
            print(line)
        print("   A B C D E")

    def game_value(self, state):
        """ Checks the current board status for a win condition

        Args:
        state (list of lists): either the current state of the game as saved in
            this TeekoPlayer object, or a generated successor state.

        Returns:
            int: 1 if this TeekoPlayer wins, -1 if the opponent wins, 0 if no winner

        TODO: complete checks for diagonal and box wins
        """
        # check horizontal wins
        for row in state:
            for i in range(2):
                if row[i] != ' ' and row[i] == row[i+1] == row[i+2] == row[i+3]:
                    return 1 if row[i]==self.my_piece else -1

        # check vertical wins
        for col in range(5):
            for i in range(2):
                if state[i][col] != ' ' and state[i][col] == state[i+1][col] == state[i+2][col] == state[i+3][col]:
                    return 1 if state[i][col]==self.my_piece else -1

        # TODO: check \ diagonal wins
        for x in range(2):
            for y in range(2):
                if state[x][y] != ' ' and state[x][y] == state[x+1][y+1] == state[x+2][y+2] == state[x+3][y+3]:
                    return 1 if state[x][y]==self.my_piece else -1
        # TODO: check / diagonal wins
        for x in range(2):
            for y in range(-2,0):
                if state[x][y] != ' ' and state[x][y] == state[x+1][y-1] == state[x+2][y-2] == state[x+3][y-3]:
                    return 1 if state[x][y]==self.my_piece else -1
        # TODO: check box wins
        for x in range(4):
            for y in range(4):
                if state[x][y] != ' ' and state[x][y] == state[x][y+1] == state[x+1][y] == state[x+1][y+1]:
                    return 1 if state[x][y] == self.my_piece else -1

        return 0 # no winner yet

    def heuristic_game_value(self, state):
        if self.game_value(state) == 1:
            return 1
        elif self.game_value(state) == -1:
            return -1
        val = 0.0
        dis, loc = self.combined_distance(state)
        for x in loc:
            val += self.perfect_weight[x[0]][x[1]]/dis
        return 1/(1+math.exp(-val))

    def combined_distance(self, state):
        curr = []
        dis = 0
        for x in range(5):
            for y in range(5):
                if state[x][y] == self.my_piece:
                    curr.append((x,y))
        if len(curr) == 1:
            return 1, curr
        npCurr = np.array(curr)
        median = np.median(npCurr, axis=0)
        for x in npCurr:
            dis+=np.linalg.norm(x-median)
        return dis/len(curr), curr #return avg distance

    def max_value(self, state, depth):
        inff = float("inf")
        alpha, beta = -inff, inff
        tVal = self.heuristic_game_value(state)
        tempState = copy.deepcopy(state)
        b = self.succ(state)

        if depth == 0 or self.game_value(state) != 0:
            return self.heuristic_game_value(state)
        for x in b:
            tempState[x[0][0]][x[0][1]] = self.my_piece
            tempState[x[1][0]][x[1][1]] = ' '
            tempVal = self.max_value(tempState, depth-1)
            alpha = max(alpha, tempVal)
            if tVal <= tempVal:
                tVal = tempVal
            if alpha >= beta:
                return alpha
            tempState = tempState = copy.deepcopy(state)
        return tVal

############################################################################
#
# THE FOLLOWING CODE IS FOR SAMPLE GAMEPLAY ONLY
#
############################################################################
def main():
    print('Hello, this is Samaritan')
    ai = TeekoPlayer()
    piece_count = 0
    turn = 0

    # drop phase
    while piece_count < 8 and ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved at "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                player_move = input("Move (e.g. B3): ")
                while player_move[0] not in "ABCDE" or player_move[1] not in "01234":
                    player_move = input("Move (e.g. B3): ")
                try:
                    ai.opponent_move([(int(player_move[1]), ord(player_move[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        piece_count += 1
        turn += 1
        turn %= 2

    # move phase - can't have a winner until all 8 pieces are on the board
    while ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved from "+chr(move[1][1]+ord("A"))+str(move[1][0]))
            print("  to "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                move_from = input("Move from (e.g. B3): ")
                while move_from[0] not in "ABCDE" or move_from[1] not in "01234":
                    move_from = input("Move from (e.g. B3): ")
                move_to = input("Move to (e.g. B3): ")
                while move_to[0] not in "ABCDE" or move_to[1] not in "01234":
                    move_to = input("Move to (e.g. B3): ")
                try:
                    ai.opponent_move([(int(move_to[1]), ord(move_to[0])-ord("A")),
                                    (int(move_from[1]), ord(move_from[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        turn += 1
        turn %= 2

    ai.print_board()
    if ai.game_value(ai.board) == 1:
        print("AI wins! Game over.")
    else:
        print("You win! Game over.")


if __name__ == "__main__":
    main()
