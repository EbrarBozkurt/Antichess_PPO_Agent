def to_coords(algebraic): # take in form 'e4' -> [4][4]
    x = algebraic[0]
    y = int(algebraic[1])
    return 8-y,('abcdefgh'.index(x))

xvals = "abcdefgh"
colours = {"w": 0,
           "b":1}

class Pawn:
    def __init__(self, colour, y: int, x: int):
        self.colour = colour
        self.x = x
        self.y = y
        self.letter = "♙"
        self.colour_letter = '♙' if colour == "b" else '♟'
        self.coordinate = f'{xvals[x]}{8-y}'
    
    def move(self,coords):
        self.x = coords[1]
        self.y = coords[0]
        self.coordinate = f'{xvals[self.x]}{8-self.y}'

    def captures(self,board):
        # Check diagonal squares
        # If is instance of an object other than empty, can move
        possibleCaptures = []
        if self.colour == "w":
            # Will move 'up' the array
            direction = -1
        elif self.colour == "b":
            # Moves 'down' the array
            direction = 1

        left = True
        right = True

        # Check if can go left
        if self.x == 0:
            left = False
        elif self.x == 7:
            right = False

        if left:
            possibleCaptures.append([self.y+direction,self.x-1]) # up left
        if right:
            possibleCaptures.append([self.y+direction,self.x+1,]) # up right
        
        captures = []
        for square in possibleCaptures:
            if not isinstance(board[square[0]][square[1]], Empty):
                if abs(colours[self.colour]-colours[board[square[0]][square[1]].colour]) == 1:
                    if square[0] == 0 or square[0] == 7: # Needs to promote
                        # Promote to king,queen,rook,knight,bishop
                        captures.append([square[0],square[1],King])
                        captures.append([square[0],square[1],Queen])
                        captures.append([square[0],square[1],Rook])
                        captures.append([square[0],square[1],Knight])
                        captures.append([square[0],square[1],Bishop])
                    else:
                        captures.append(square)
        
        return captures

    def moves(self, board):
        moves = []
        if self.colour == "w":
            # Will move 'up' the array
            direction = -1
        elif self.colour == "b":
            # Moves 'down' the array
            direction = 1

        # Always can move one square forward
        # On 8th or 1st rank, it will be promoted
        # Check if piece blocking
        if isinstance(board[self.y+direction][self.x], Empty):
            if self.y + direction == 0 or self.y + direction == 7: # Needs to promote
                # Promote to king,queen,rook,knight,bishop
                moves.append([self.y+direction,self.x,King])
                moves.append([self.y+direction,self.x,Queen])
                moves.append([self.y+direction,self.x,Rook])
                moves.append([self.y+direction,self.x,Knight])
                moves.append([self.y+direction,self.x,Bishop])
            else:
                moves.append([self.y+direction,self.x])

        # or 2 squares if it hasn't moved yet
        if self.colour == "w":
            # If on second rank and white
            if self.y == 6: # 8-2
                if isinstance(board[self.y+direction][self.x], Empty):
                    if isinstance(board[self.y+(direction*2)][self.x], Empty):
                        moves.append([self.y+(direction*2),self.x])
        elif self.colour == "b":
            # If on seventh rank and black
            if self.y == 1: # 8-1
                if isinstance(board[self.y+direction][self.x], Empty):
                    if isinstance(board[self.y+(direction*2)][self.x], Empty):
                        moves.append([self.y+(direction*2),self.x])

        return moves

    def enpassant(self,chessBoard,enpassant):
        if enpassant == '-':
            return [[[]]]
        y,x = to_coords(enpassant)

        if self.colour == "w":
            # Will move 'up' the array
            direction = -1
        elif self.colour == "b":
            # Moves 'down' the array
            direction = 1
        
        # Must be 1x val away and 1 y in direction of travel
        if abs(self.x-x) == 1 and self.y+direction == y:
            return [[[self.y,self.x],[self.y,x]],[[self.y,x],[y,x]]]
        return [[[]]]

class Rook:
    def __init__(self, colour, y: int,x: int):
        self.colour = colour
        self.x = x
        self.y = y
        self.letter = "♖"
        self.colour_letter = '♖' if colour == "b" else '♜'
        self.coordinate = f'{xvals[x]}{8-y}'

    def move(self,coords):
        self.x = coords[1]
        self.y = coords[0]
        self.coordinate = f'{xvals[self.x]}{8-self.y}'

    def captures(self,board):
        possibleCaps = []
        # Find number of squares up before a piece, or end of board
        # y value is number of squares away from the top
        for i in range(1,self.y+1):
            if not isinstance(board[self.y-i][self.x], Empty): # There is something there
                if abs(colours[self.colour]-colours[board[self.y-i][self.x].colour]) == 1:
                    possibleCaps.append([self.y-i,self.x])
                break

        # Go down
        # 8-(y+1) is num squares from bottom
        for i in range(1,8-self.y):
            if not isinstance(board[self.y+i][self.x], Empty): # There is something there
                if abs(colours[self.colour]-colours[board[self.y+i][self.x].colour]) == 1:
                    possibleCaps.append([self.y+i,self.x])
                break

        # Go right
        # 8-(y+1) is num squares from right
        for i in range(1,8-self.x):
            if not isinstance(board[self.y][self.x+i], Empty): # There is something there
                if abs(colours[self.colour]-colours[board[self.y][self.x+i].colour]) == 1:
                    possibleCaps.append([self.y,self.x+i])
                break

        # Go left
        # y-1 is num squares from right
        for i in range(1,self.x+1):
            if not isinstance(board[self.y][self.x-i], Empty): # There is something there
                if abs(colours[self.colour]-colours[board[self.y][self.x-i].colour]) == 1:
                    possibleCaps.append([self.y,self.x-i])
                break

        return possibleCaps

    def moves(self,board):
        moves = []
        
        # Find all possible moves now
        for i in range(1,self.y+1):
            if isinstance(board[self.y-i][self.x], Empty): # Empty space, can move
                moves.append([self.y-i,self.x])
            else: # There's something there
                break

        # Go down
        for i in range(1,8-self.y):
            if isinstance(board[self.y+i][self.x], Empty): # Empty space, can move
                moves.append([self.y+i,self.x])
            else: # There's something there
                break

        # Go right
        for i in range(1,8-self.x):
            if isinstance(board[self.y][self.x+i], Empty): # Empty space, can move
                moves.append([self.y,self.x+i])
            else: # There's something there
                break

        # Go left
        for i in range(1,self.x+1):
            if isinstance(board[self.y][self.x-i], Empty): # Empty space, can move
                moves.append([self.y,self.x-i])
            else: # There's something there
                break

        return moves
            
class Knight:
    def __init__(self, colour, y: int,x: int):
        self.colour = colour
        self.x = x
        self.y = y
        self.letter = "♘"
        self.colour_letter = '♘' if colour == "b" else '♞'
        self.coordinate = f'{xvals[x]}{8-y}'

    def move(self,coords):
        self.x = coords[1]
        self.y = coords[0]
        self.coordinate = f'{xvals[self.x]}{8-self.y}'

    def captures(self,board):
        possibleMoves = []
        captureMoves = []

        if self.y-2 >= 0:
            if self.x-1 >= 0:
                possibleMoves.append([self.y-2,self.x-1])
            if self.x+1 <= 7:
                possibleMoves.append([self.y-2,self.x+1])
        if self.y+2 <= 7:
            if self.x-1 >= 0:
                possibleMoves.append([self.y+2,self.x-1])
            if self.x+1 <= 7:
                possibleMoves.append([self.y+2,self.x+1])
        if self.x-2 >= 0:
            if self.y-1 >= 0:
                possibleMoves.append([self.y-1,self.x-2])
            if self.y+1 <= 7:
                possibleMoves.append([self.y+1,self.x-2])
        if self.x+2 <= 7:
            if self.y-1 >= 0:
                possibleMoves.append([self.y-1,self.x+2])
            if self.y+1 <= 7:
                possibleMoves.append([self.y+1,self.x+2])

        # Test which are captures
        for move in possibleMoves:
            if not isinstance(board[move[0]][move[1]], Empty):
                if abs(colours[self.colour]-colours[board[move[0]][move[1]].colour]) == 1:
                    captureMoves.append(move)
        return captureMoves
    
    def moves(self,board):
        possibleMoves = []
        emptyMoves = []

        if self.y-2 >= 0:
            if self.x-1 >= 0:
                possibleMoves.append([self.y-2,self.x-1])
            if self.x+1 <= 7:
                possibleMoves.append([self.y-2,self.x+1])
        if self.y+2 <= 7:
            if self.x-1 >= 0:
                possibleMoves.append([self.y+2,self.x-1])
            if self.x+1 <= 7:
                possibleMoves.append([self.y+2,self.x+1])
        if self.x-2 >= 0:
            if self.y-1 >= 0:
                possibleMoves.append([self.y-1,self.x-2])
            if self.y+1 <= 7:
                possibleMoves.append([self.y+1,self.x-2])
        if self.x+2 <= 7:
            if self.y-1 >= 0:
                possibleMoves.append([self.y-1,self.x+2])
            if self.y+1 <= 7:
                possibleMoves.append([self.y+1,self.x+2])

        # Test which are captures
        for move in possibleMoves:
            if isinstance(board[move[0]][move[1]], Empty):
                emptyMoves.append(move)
        return emptyMoves
        
class Bishop:
    def __init__(self, colour, y: int,x: int):
        self.colour = colour
        self.x = x
        self.y = y
        self.letter = "♗"
        self.colour_letter = '♗' if colour == "b" else '♝'
        self.coordinate = f'{xvals[x]}{8-y}'

    def move(self,coords):
        self.x = coords[1]
        self.y = coords[0]
        self.coordinate = f'{xvals[self.x]}{8-self.y}'

    def captures(self,board):
        possibleCaps = []
        # Num. squares to the top right is min of top and right
        for i in range(1,min(self.y,7-self.x)+1):
            if not isinstance(board[self.y-i][self.x+i], Empty): # There is something there
                if abs(colours[self.colour]-colours[board[self.y-i][self.x+i].colour]) == 1:
                    possibleCaps.append([self.y-i,self.x+i])
                break
    
        # Num.squares to top left
        for i in range(1,min(self.y,self.x)+1):
            if not isinstance(board[self.y-i][self.x-i], Empty): # There is something there
                if abs(colours[self.colour]-colours[board[self.y-i][self.x-i].colour]) == 1:
                    possibleCaps.append([self.y-i,self.x-i])
                break

        # Num.squares to bottom left
        for i in range(1,min(7-self.y,self.x)+1):
            if not isinstance(board[self.y+i][self.x-i], Empty): # There is something there
                if abs(colours[self.colour]-colours[board[self.y+i][self.x-i].colour]) == 1:
                    possibleCaps.append([self.y+i,self.x-i])
                break

        # Num.squares to bottom right
        for i in range(1,min(7-self.y,7-self.x)+1):
            if not isinstance(board[self.y+i][self.x+i], Empty): # There is something there
                if abs(colours[self.colour]-colours[board[self.y+i][self.x+i].colour]) == 1:
                    possibleCaps.append([self.y+i,self.x+i])
                break

        return possibleCaps

    def moves(self,board):
        moves = []

        for i in range(1,min(self.y,7-self.x)+1):
            if not isinstance(board[self.y-i][self.x+i], Empty): # There is something there
                break
            moves.append([self.y-i,self.x+i])
    
        # Num.squares to top left
        for i in range(1,min(self.y,self.x)+1):
            if not isinstance(board[self.y-i][self.x-i], Empty): # There is something there
                break
            moves.append([self.y-i,self.x-i])

        # Num.squares to bottom left
        for i in range(1,min(7-self.y,self.x)+1):
            if not isinstance(board[self.y+i][self.x-i], Empty): # There is something there
                break
            moves.append([self.y+i,self.x-i])

        # Num.squares to bottom right
        for i in range(1,min(7-self.y,7-self.x)+1):
            if not isinstance(board[self.y+i][self.x+i], Empty): # There is something there
                break
            moves.append([self.y+i,self.x+i])

        return moves

class King:
    def __init__(self, colour, y: int,x: int):
        self.colour = colour
        self.x = x
        self.y = y
        self.letter = "♔"
        self.colour_letter = '♔' if colour == "b" else '♚'
        self.coordinate = f'{xvals[x]}{8-y}'

    def move(self,coords):
        self.x = coords[1]
        self.y = coords[0]
        self.coordinate = f'{xvals[self.x]}{8-self.y}'

    def captures(self,board):
        possibleCaps = []
        captures = []
        # Can go upleft,up,upright,right,downright,down,downleft,left
        if self.y-1 >= 0: # Can go up
            possibleCaps.append([self.y-1,self.x])
            if self.x-1 >= 0:
                possibleCaps.append([self.y-1,self.x-1])
                possibleCaps.append([self.y,self.x-1]) # Can also just go left
            if self.x+1 <= 7:
                possibleCaps.append([self.y-1,self.x+1])
                possibleCaps.append([self.y,self.x+1])
        if self.y+1 <= 7: # Can go down
            possibleCaps.append([self.y+1,self.x])
            if self.x-1 >= 0:
                possibleCaps.append([self.y+1,self.x-1])
            if self.x+1 <= 7:
                possibleCaps.append([self.y+1,self.x+1])

        for move in possibleCaps:
            if not isinstance(board[move[0]][move[1]], Empty):
                if abs(colours[self.colour]-colours[board[move[0]][move[1]].colour]) == 1:
                    captures.append(move)

        return captures

    def moves(self,board):
        possibleMoves = []
        moves = []
        if self.y-1 >= 0: # Can go up
            possibleMoves.append([self.y-1,self.x])
            if self.x-1 >= 0:
                possibleMoves.append([self.y-1,self.x-1])
                possibleMoves.append([self.y,self.x-1]) # Can also just go left
            if self.x+1 <= 7:
                possibleMoves.append([self.y-1,self.x+1])
                possibleMoves.append([self.y,self.x+1])
        if self.y+1 <= 7: # Can go down
            possibleMoves.append([self.y+1,self.x])
            if self.x-1 >= 0:
                possibleMoves.append([self.y+1,self.x-1])
            if self.x+1 <= 7:
                possibleMoves.append([self.y+1,self.x+1])

        for move in possibleMoves:
            if isinstance(board[move[0]][move[1]], Empty):
                moves.append(move)

        return moves

    # def castling(self,board,castling):
    #     # castling - KQkq
    #     moves = []
    #     if self.colour == "w":
    #         if "K" in castling: # Kingside castling with white
    #             # King is on 7,4
    #             # Rook on 7,7
    #             # Check for pieces on 7,5 , 7,6
    #             if isinstance(board[7][5], Empty):
    #                 if isinstance(board[7][6], Empty):
    #                     moves.append([[[7,4],[7,6]],[[7,7],[7,5]]])
    #         if "Q" in castling: # Queenside castling with white
    #             # King is on 7,4
    #             # Rook on 7,0
    #             # Check for pieces on 7,3 , 7,2 , 7,1
    #             if isinstance(board[7][3], Empty):
    #                 if isinstance(board[7][2], Empty):
    #                     if isinstance(board[7][1], Empty):
    #                         moves.append([[[7,4],[7,2]],[[7,0],[7,3]]])
    #     elif self.colour == "b":
    #         if "k" in castling: # Kingside castling with white
    #             if isinstance(board[0][4], Empty):
    #                 if isinstance(board[0][7], Empty):
    #                     moves.append([[[0,4],[0,6]],[[0,7],[0,5]]])
    #         if "q" in castling: # Queenside castling with white
    #             if isinstance(board[0][3], Empty):
    #                 if isinstance(board[0][2], Empty):
    #                     if isinstance(board[0][1], Empty):
    #                         moves.append([[[0,4],[0,2]],[[0,0],[0,3]]])
        
    #     return moves

class Queen:
    def __init__(self, colour, y: int,x: int):
        self.colour = colour
        self.x = x
        self.y = y
        self.letter = "♕"
        self.colour_letter = '♕' if colour == "b" else '♛'
        self.coordinate = f'{xvals[x]}{8-y}'
    
    def move(self,coords):
        self.x = coords[1]
        self.y = coords[0]
        self.coordinate = f'{xvals[self.x]}{8-self.y}'

    def captures(self,board): # Literally copy paste rook and bishopt together
        possibleCaps = []
        # Find number of squares up before a piece, or end of board
        # y value is number of squares away from the top
        for i in range(1,self.y+1):
            if not isinstance(board[self.y-i][self.x], Empty): # There is something there
                if abs(colours[self.colour]-colours[board[self.y-i][self.x].colour]) == 1:
                    possibleCaps.append([self.y-i,self.x])
                break

        # Go down
        # 8-(y+1) is num squares from bottom
        for i in range(1,8-self.y):
            if not isinstance(board[self.y+i][self.x], Empty): # There is something there
                if abs(colours[self.colour]-colours[board[self.y+i][self.x].colour]) == 1:
                    possibleCaps.append([self.y+i,self.x])
                break

        # Go right
        # 8-(y+1) is num squares from right
        for i in range(1,8-self.x):
            if not isinstance(board[self.y][self.x+i], Empty): # There is something there
                if abs(colours[self.colour]-colours[board[self.y][self.x+i].colour]) == 1:
                    possibleCaps.append([self.y,self.x+i])
                break

        # Go left
        # y-1 is num squares from right
        for i in range(1,self.x+1):
            if not isinstance(board[self.y][self.x-i], Empty): # There is something there
                if abs(colours[self.colour]-colours[board[self.y][self.x-i].colour]) == 1:
                    possibleCaps.append([self.y,self.x-i])
                break

        for i in range(1,min(self.y,7-self.x)+1):
            if not isinstance(board[self.y-i][self.x+i], Empty): # There is something there
                if abs(colours[self.colour]-colours[board[self.y-i][self.x+i].colour]) == 1:
                    possibleCaps.append([self.y-i,self.x+i])
                break
    
        # Num.squares to top left
        for i in range(1,min(self.y,self.x)+1):
            if not isinstance(board[self.y-i][self.x-i], Empty): # There is something there
                if abs(colours[self.colour]-colours[board[self.y-i][self.x-i].colour]) == 1:
                    possibleCaps.append([self.y-i,self.x-i])
                break

        # Num.squares to bottom left
        for i in range(1,min(7-self.y,self.x)+1):
            if not isinstance(board[self.y+i][self.x-i], Empty): # There is something there
                if abs(colours[self.colour]-colours[board[self.y+i][self.x-i].colour]) == 1:
                    possibleCaps.append([self.y+i,self.x-i])
                break

        # Num.squares to bottom right
        for i in range(1,min(7-self.y,7-self.x)+1):
            if not isinstance(board[self.y+i][self.x+i], Empty): # There is something there
                if abs(colours[self.colour]-colours[board[self.y+i][self.x+i].colour]) == 1:
                    possibleCaps.append([self.y+i,self.x+i])
                break

        return possibleCaps

    def moves(self,board):
        moves = []

        for i in range(1,min(self.y,7-self.x)+1):
            if not isinstance(board[self.y-i][self.x+i], Empty): # There is something there
                break
            moves.append([self.y-i,self.x+i])
    
        # Num.squares to top left
        for i in range(1,min(self.y,self.x)+1):
            if not isinstance(board[self.y-i][self.x-i], Empty): # There is something there
                break
            moves.append([self.y-i,self.x-i])

        # Num.squares to bottom left
        for i in range(1,min(7-self.y,self.x)+1):
            if not isinstance(board[self.y+i][self.x-i], Empty): # There is something there
                break
            moves.append([self.y+i,self.x-i])

        # Num.squares to bottom right
        for i in range(1,min(7-self.y,7-self.x)+1):
            if not isinstance(board[self.y+i][self.x+i], Empty): # There is something there
                break
            moves.append([self.y+i,self.x+i])
        
        # Find all possible moves now
        for i in range(1,self.y+1):
            if isinstance(board[self.y-i][self.x], Empty): # Empty space, can move
                moves.append([self.y-i,self.x])
            else: # There's something there
                break

        # Go down
        for i in range(1,8-self.y):
            if isinstance(board[self.y+i][self.x], Empty): # Empty space, can move
                moves.append([self.y+i,self.x])
            else: # There's something there
                break

        # Go right
        for i in range(1,8-self.x):
            if isinstance(board[self.y][self.x+i], Empty): # Empty space, can move
                moves.append([self.y,self.x+i])
            else: # There's something there
                break

        # Go left
        for i in range(1,self.x+1):
            if isinstance(board[self.y][self.x-i], Empty): # Empty space, can move
                moves.append([self.y,self.x-i])
            else: # There's something there
                break

        return moves

class Empty:
    def __init__(self):
        self.colour = '-'
        self.x = '-'
        self.y = '-'
        self.letter = "-"
        self.colour_letter = '-'
        self.coordinate = f'--'

    def captures(self,board):
        return []
    
    def moves(self,board):
        return []