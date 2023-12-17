import pieces
import re

def print_board(board: list):
    output = [[],[],[],[],[],[],[],[]]
    for y,rank in enumerate(board):
        for x,square in enumerate(rank):
            output[y].append(getattr(board[y][x],"colour_letter"))

    # board done
    # now: print fancy
    outstr = ''
    for rank in output:
        for square in rank:
            outstr += f'{square} '
        outstr += "\n"
    
    return outstr

def to_coords(algebraic): # take in form 'e4' -> [4][4]
    x = algebraic[0]
    y = int(algebraic[1])
    return 8-y,('abcdefgh'.index(x))

def from_coords(xy):
    return f"{'abcdefgh'[xy[1]]}{8-xy[0]}"

def get_coords(ords,chessBoard):
    return chessBoard[ords[0]][ords[1]]

def make_move(piece,move,chessBoard): # Update enpassant here
    # Check if piece is pawn
    # Check if it moves two squares
    # Remove enpassant from previous
    if piece.colour == "w":
        # Will move 'up' the array
        direction = -1
    elif piece.colour == "b":
        # Moves 'down' the array
        direction = 1
    
    enpassant = '-'
    if isinstance(piece,pieces.Pawn):
        if piece.y + direction*2 == move[0]:
            enpassant = from_coords([piece.y+direction,piece.x])
    
    chessBoard[piece.y][piece.x] = pieces.Empty()
    piece.move(move)
    if len(move) == 3:
        chessBoard[move[0]][move[1]] = move[2](piece.colour,move[0],move[1])
    else:
        chessBoard[move[0]][move[1]] = piece

    return chessBoard,enpassant

def find_moves(chessBoard, whosMove, enpassant):
    legalMoves = []
    canCapture = False
    for row in chessBoard: # Determine whether you can capture
        for piece in row:
            if piece.colour == whosMove:
                if piece.captures(chessBoard) != []:
                    canCapture = True
                    break
                if isinstance(piece, pieces.Pawn):
                    if piece.enpassant(chessBoard,enpassant) != [[[]]]:
                        canCapture = True
                        break

    if canCapture: # Can capture
        for row in chessBoard:
            for piece in row:
                if piece.colour == whosMove:
                    if piece.captures(chessBoard) != []:
                        for move in piece.captures(chessBoard):
                            legalMoves.append([[[piece.y,piece.x],move]])
                    if isinstance(piece, pieces.Pawn):
                        if piece.enpassant(chessBoard,enpassant) != [[[]]]:
                            legalMoves.append(piece.enpassant(chessBoard,enpassant))
    else: # Can't capture
        for row in chessBoard:
            for piece in row:
                if piece.colour == whosMove:
                    if piece.moves(chessBoard) != []:
                        for move in piece.moves(chessBoard):
                            legalMoves.append([[[piece.y,piece.x],move]])
                    # if isinstance(piece,pieces.King):
                    #     for move in piece.castling(chessBoard,castling):
                    #         legalMoves.append(move)

    return legalMoves

def checkmate(chessBoard,whosMove):
    # Determine whether the game has been won
    # The other player can only checkmate you, check for player who's turn it is before they move

    # Say it's white's move
    hasPieces = False
    hasMoves = False
    winner = False
    for row in chessBoard:
        for piece in row:
            if piece.colour == whosMove: # If white has a piece, it has not lost
                hasPieces = True
                if piece.captures(chessBoard) != [] or piece.moves(chessBoard) != []:
                    hasMoves = True

    if not hasPieces or not hasMoves:
        winner = whosMove
    
    return winner

def convert_user_coords(coords,chessBoard,whosMove,EnPassant):
    pieceLetters = {
        "p": pieces.Pawn,
        "r": pieces.Rook,
        "n": pieces.Knight,
        "b": pieces.Bishop,
        "q": pieces.Queen,
        "k": pieces.King
    }

    # will be algebraic - e2e4
    # attach =Q on the end for promotions
    if re.match('[a-h]\d[a-h]\d',coords):
        move = [[list(to_coords(coords[:2])),list(to_coords(coords[2:4]))]]
        if coords[2:4] == EnPassant:
            move = [[list(to_coords(coords[:2])),[to_coords(coords[:2])[0],to_coords(coords[2:4])[1]]],[[to_coords(coords[:2])[0],to_coords(coords[2:4])[1]],list(to_coords(coords[2:4]))]]
        if coords[-2] == "=":
            move = [[list(to_coords(coords[:2])),[to_coords(coords[2:4])[0],to_coords(coords[2:4])[1],pieceLetters[coords[-1].lower()]]]]
    else:
        return False
    return move

# def castling_rights(chessBoard,whosMove,castling):
#     # whosMove has not been changed yet, determine whether they have lost castling rights
#     # If king has moved
#     if not isinstance(chessBoard[7][4],pieces.King):
#         castling = castling.replace("K", '')
#         castling = castling.replace("Q", '')
#     if not isinstance(chessBoard[7][7],pieces.Rook):
#         castling = castling.replace("K", '')
#     if not isinstance(chessBoard[7][0],pieces.Rook):
#         castling = castling.replace("Q", '')
#     if not isinstance(chessBoard[0][4],pieces.King):
#         castling = castling.replace("k", '')
#         castling = castling.replace("q", '')
#     if not isinstance(chessBoard[0][7],pieces.Rook):
#         castling = castling.replace("k", '')
#     if not isinstance(chessBoard[0][0],pieces.Rook):
#         castling = castling.replace("q", '')

#     return castling
