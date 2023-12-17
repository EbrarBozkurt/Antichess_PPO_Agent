import pieces

def setup_fen(FENstr):

    pieceLetters = {
        "p": pieces.Pawn,
        "r": pieces.Rook,
        "n": pieces.Knight,
        "b": pieces.Bishop,
        "q": pieces.Queen,
        "k": pieces.King
    }

    rows = FENstr.split("/")
    others = rows[-1].split(" ")[1:]
    WhosMove = others[0]
    Castling = others[1]
    EnPassant = others[2]
    # ignore half move clock
    Fullmove = others[4]
    rows[-1] = rows[-1].split(" ")[0]

    empty = pieces.Empty()

    start_board = [
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        []]
    
    for index,row in enumerate(rows): 
        for char in row:
            if char.isdigit():
                for i in range(int(char)):
                    start_board[index].append(empty)
            elif char.upper() == char: # Is uppercase (white piece)
                start_board[index].append(pieceLetters[char.lower()]("w",index,len(start_board[index])))
            elif char.lower() == char: # Is uppercase (white piece)
                start_board[index].append(pieceLetters[char]("b",index,len(start_board[index])))

    return start_board, WhosMove, Castling, EnPassant, Fullmove

def export_fen(board):
    empties = 0
    FENstr = ''
    for row in board:
        for piece in row:
            if isinstance(piece, pieces.Empty):
                empties += 1
            else:
                if empties > 0:
                    FENstr += str(empties)
                    empties = 0
                if piece.colour == "w":
                    FENstr += piece.letter
                elif piece.colour == "b":
                    FENstr += piece.letter.lower()
        if empties > 0:
            FENstr += str(empties)
            empties = 0
        FENstr += "/"
    return FENstr[:-1]
    # 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'