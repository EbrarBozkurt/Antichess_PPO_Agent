import fen_setup
import board
import ai
import ppoy
import pieces
import torch
import numpy as np
import aix
# Start setup is 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'

exports = fen_setup.setup_fen('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
chessBoard = exports[0]
whosMove = exports[1]
# castling = exports[2]
EnPassant = exports[3]
stop = False
move = False


def numeric_to_algebraic(numeric):
    # Define a reverse mapping of numerical values to chess squares
    #print("numeric",numeric)
    start_numeric = int(numeric/64)
    end_numeric = numeric%64
    reverse_mapping = {
        0: 'a1', 8: 'b1', 16: 'c1', 24: 'd1', 32: 'e1', 40: 'f1', 48: 'g1', 56: 'h1',
        1: 'a2', 9: 'b2', 17: 'c2', 25: 'd2', 33: 'e2', 41: 'f2', 49: 'g2', 57: 'h2',
        2: 'a3', 10: 'b3', 18: 'c3', 26: 'd3', 34: 'e3', 42: 'f3', 50: 'g3', 58: 'h3',
        3: 'a4', 11: 'b4', 19: 'c4', 27: 'd4', 35: 'e4', 43: 'f4', 51: 'g4', 59: 'h4',
        4: 'a5', 12: 'b5', 20: 'c5', 28: 'd5', 36: 'e5', 44: 'f5', 52: 'g5', 60: 'h5',
        5: 'a6', 13: 'b6', 21: 'c6', 29: 'd6', 37: 'e6', 45: 'f6', 53: 'g6', 61: 'h6',
        6: 'a7', 14: 'b7', 22: 'c7', 30: 'd7', 38: 'e7', 46: 'f7', 54: 'g7', 62: 'h7',
        7: 'a8', 15: 'b8', 23: 'c8', 31: 'd8', 39: 'e8', 47: 'f8', 55: 'g8', 63: 'h8',
    }
    # Convert numeric values to algebraic notation
    start_square = reverse_mapping[start_numeric]
    end_square = reverse_mapping[end_numeric]
    #x = string(start_square)+string(end_square)
    

    return start_square + end_square


def convert_chessboard_to_string(chessBoard):
    boardString = ''
    i = 0
    for row in chessBoard:
        for piece in row:
            i += 1
            #print(i)
            #if piece is pawn, add 1
            if type(piece) == pieces.Pawn and piece.colour == 'w':
                boardString += 'p'
            elif type(piece) == pieces.Pawn and piece.colour == 'b':
                boardString += 'P'
            #if piece is rook, add 2
            elif type(piece) == pieces.Rook and piece.colour == 'w':
                boardString += 'r'
            elif type(piece) == pieces.Rook and piece.colour == 'b':
                boardString += 'R'
            #if piece is knight, add 3
            elif type(piece) == pieces.Knight and piece.colour == 'w':
                boardString += 'n'
            elif type(piece) == pieces.Knight and piece.colour == 'b':
                boardString += 'N'
            #if piece is bishop, add 4
            elif type(piece) == pieces.Bishop and piece.colour == 'w':
                boardString += 'b'
            elif type(piece) == pieces.Bishop and piece.colour == 'b':
                boardString += 'B'
            #if piece is queen, add 5
            elif type(piece) == pieces.Queen and piece.colour == 'w':
                boardString += 'q'
            elif type(piece) == pieces.Queen and piece.colour == 'b':
                boardString += 'Q'
            #if piece is king, add 6
            elif type(piece) == pieces.King and piece.colour == 'w':
                boardString += 'k'
            elif type(piece) == pieces.King and piece.colour == 'b':
                boardString += 'K'
            #if piece is empty, add 0
            elif piece == 0:
                boardString += '0'
    return boardString



#convert chessboard to array
def convert_chessboard_to_array(chessBoard):
    boardArray = []
    for row in chessBoard:
        for piece in row:
            #if piece is pawn, add 1
            if type(piece) == pieces.Pawn and piece.colour == 'w':
                boardArray.append(1)
            elif type(piece) == pieces.Pawn and piece.colour == 'b':
                boardArray.append(7)
            #if piece is rook, add 2
            elif type(piece) == pieces.Rook and piece.colour == 'w':
                boardArray.append(2)
            elif type(piece) == pieces.Rook and piece.colour == 'b':
                boardArray.append(8)
            #if piece is knight, add 3
            elif type(piece) == pieces.Knight and piece.colour == 'w':
                boardArray.append(3)
            elif type(piece) == pieces.Knight and piece.colour == 'b':
                boardArray.append(9)
            #if piece is bishop, add 4
            elif type(piece) == pieces.Bishop and piece.colour == 'w':
                boardArray.append(4)
            elif type(piece) == pieces.Bishop and piece.colour == 'b':
                boardArray.append(10)
            #if piece is queen, add 5
            elif type(piece) == pieces.Queen and piece.colour == 'w':
                boardArray.append(5)
            elif type(piece) == pieces.Queen and piece.colour == 'b':
                boardArray.append(11)
            #if piece is king, add 6
            elif type(piece) == pieces.King and piece.colour == 'w':
                boardArray.append(6)
            elif type(piece) == pieces.King and piece.colour == 'b':
                boardArray.append(12)
            #if piece is empty, add 0
            elif piece == 0:
                boardArray.append(0)
    return boardArray

#convert chessboard to numpy array
def convert_chessboard_to_numpy(chessBoard):
    i = 0
    boardArray = np.array([])
    for row in chessBoard:
        for piece in row:
            i+=1
            #print(i)
            #if piece is pawn, add 1
            if type(piece) == pieces.Pawn and piece.colour == 'w':
                boardArray = np.append(boardArray,1)
            elif type(piece) == pieces.Pawn and piece.colour == 'b':
                boardArray = np.append(boardArray,7)
            #if piece is rook, add 2
            elif type(piece) == pieces.Rook and piece.colour == 'w':
                boardArray = np.append(boardArray,2)
            elif type(piece) == pieces.Rook and piece.colour == 'b':
                boardArray = np.append(boardArray,8)
            #if piece is knight, add 3
            elif type(piece) == pieces.Knight and piece.colour == 'w':
                boardArray = np.append(boardArray,3)
            elif type(piece) == pieces.Knight and piece.colour == 'b':
                boardArray = np.append(boardArray,9)
            #if piece is bishop, add 4
            elif type(piece) == pieces.Bishop and piece.colour == 'w':
                boardArray = np.append(boardArray,4)
            elif type(piece) == pieces.Bishop and piece.colour == 'b':
                boardArray = np.append(boardArray,10)
            #if piece is queen, add 5
            elif type(piece) == pieces.Queen and piece.colour == 'w':
                boardArray = np.append(boardArray,5)
            elif type(piece) == pieces.Queen and piece.colour == 'b':
                boardArray = np.append(boardArray,11)
            #if piece is king, add 6
            elif type(piece) == pieces.King and piece.colour == 'w':
                boardArray = np.append(boardArray,6)
            elif type(piece) == pieces.King and piece.colour == 'b':
                boardArray = np.append(boardArray,12)
            #if piece is empty, add 0
            elif type(piece)==pieces.Empty:
                boardArray = np.append(boardArray,0)
    #print("boardArray",boardArray)
    return boardArray







def play_random_game(legalMoves, chessBoard):
    return ai.ai(legalMoves,chessBoard)

def play_human_game(legalMoves, chessBoard):
    move = False
    while move == False:
        move = board.convert_user_coords(input('>> '),chessBoard,whosMove,EnPassant)
        if move not in legalMoves:
            move = False
        if move == False:
            print("That's an illegal move. Try again.")
    return move


def play_ppo_agent_game(legalMoves, chessBoard):
    return False

def play_game():
    while not stop:
        #print(board.print_board(chessBoard))
        legalMoves = board.find_moves(chessBoard,whosMove,EnPassant)
        # Computer plays black
        if whosMove == "b":
            move =  play_random_game(legalMoves, chessBoard)
        else:
            move = play_human_game(legalMoves, chessBoard)
        for moves in move:
            chessBoard, EnPassant = board.make_move(chessBoard[moves[0][0]][moves[0][1]],moves[1],chessBoard)
        move = False
        # if castling != '':
        #     castling = board.castling_rights(chessBoard,whosMove,castling)
        whosMove = "w" if whosMove == "b" else "b" # Swap
        if board.checkmate(chessBoard,whosMove) != False:
            print(board.print_board(chessBoard))
            winner = 'White' if board.checkmate(chessBoard,whosMove) == "w" else 'Black'
            print(f'{winner} has won the game!')
            stop = True


# Set your environment parameters
input_size = 64  # Assuming a flat representation of the chess board as input
output_size = 64*64  # Number of legal moves in your chess environment

# Initialize the PPO agent
agent = ppoy.PPOAgent(input_size, output_size)

# Training loop
num_episodes = 100
ppo_epochs = 4
x = 0
for episode in range(num_episodes):
    #x = x  + 1
    print("episode: ", episode)
    
    exports = fen_setup.setup_fen('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
    chessBoard = exports[0]
    whosMove = exports[1]
    # castling = exports[2]
    EnPassant = exports[3]
    stop = False
    move = False

    state =convert_chessboard_to_numpy(chessBoard)
    #print("/state",state)
    done = False

    #states, actions, rewards, log_probs, values, next_values, dones = [], [], [], [], [], [], []
    x=0
    #stop = False
    while not stop:
        x = x + 1
        #print("state",state)
        
        #next_state, reward, done, _ = env.step(action)
        reward = 0
        print(board.print_board(chessBoard))
        legalMoves = board.find_moves(chessBoard,whosMove,EnPassant)
        if len(legalMoves) == 0:
            break

        # Computer plays black
        if whosMove == "b":
            move = aix.ai(legalMoves,chessBoard,EnPassant,whosMove)
            #move =  play_random_game(legalMoves, chessBoard)
        else:
            move = False
            while move == False:
                action, log_prob = agent.select_action(state)
                actionx = numeric_to_algebraic(action)
                #print("actionx",actionx)
                move = board.convert_user_coords(actionx ,chessBoard,whosMove,EnPassant)
                if move not in legalMoves:
                    move = False
                else:
                    print("action",actionx)
                #if move == False:
                #    print("That's an illegal move. Try again.")
            #print("move",move)
        for moves in move:
            chessBoard, EnPassant = board.make_move(chessBoard[moves[0][0]][moves[0][1]],moves[1],chessBoard)
            if EnPassant != '-' and whosMove == 'w':
                reward = 1
            elif EnPassant != '-' and whosMove == 'b':
                reward = -1
        next_state = convert_chessboard_to_numpy(chessBoard)
        move = False
        # if castling != '':
        #     castling = board.castling_rights(chessBoard,whosMove,castling)
        whosMove = "w" if whosMove == "b" else "b" # Swap
        if board.checkmate(chessBoard,whosMove) != False:
            print(board.print_board(chessBoard))
            winner = 'White' if board.checkmate(chessBoard,whosMove) == "w" else 'Black'
            print(f'{winner} has won the game!')
            if winner == 'Black':
                reward = -16
            else:
                reward = 16
            stop = True


        