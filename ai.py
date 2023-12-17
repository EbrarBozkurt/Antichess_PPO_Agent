import random
import Node
import board
import copy

def ai(moves,chessBoard):
    #print("moves: ",moves)
    return random.choice(moves)
   
def selection(curr_node, playColor, whosMove):
    if(not curr_node.children):
        return curr_node
    
    checkmated = True
    for child in curr_node.children:
        if child.checkmated == False:
            checkmated = False
            break

    if checkmated:
        curr_node.checkmated = True
        whosMove = "w" if whosMove == "b" else "b" # Swap
        return selection(curr_node.parent, playColor, whosMove)

    if(playColor == whosMove):
        max_ucb = -100
        selected_child = None
        for i in curr_node.children:
            if i.checkmated == False:
                curr_ucb = i.data
                if(curr_ucb > max_ucb):
                    max_ucb = curr_ucb
                    selected_child = i
    else:
        min_ucb = 100
        selected_child = None
        for i in curr_node.children:
            if i.checkmated == False:
                curr_ucb = i.data
                if(curr_ucb < min_ucb):
                    min_ucb = curr_ucb
                    selected_child = i

    whosMove = "w" if whosMove == "b" else "b" # Swap
    
    return selection(selected_child, playColor, whosMove)

def expansion(curr_node, moves, chessBoard: list, EnPassant, whosMove):
    newNode = None
    newNode2 = None
    prevNode = None
    whosMoveReverse = "w" if whosMove == "b" else "b" # Swap
    for move in moves:
        newBoard = copy.deepcopy(chessBoard)

        for moves1 in move:
            newBoard, EnPassant = board.make_move(newBoard[moves1[0][0]][moves1[0][1]],moves1[1],newBoard)
        
        newNode = Node.Node(newBoard, whosMoveReverse, move)
        curr_node.children.append(newNode)
        newNode.parent = curr_node

        winlet = board.checkmate(newBoard,whosMoveReverse)
        if winlet != False:
            if winlet == whosMove:
                newNode.data = 1
                curr_node.data = 4 * (curr_node.data / 5) + 1 / 5
            elif winlet == whosMoveReverse:
                newNode.data = -1
                curr_node.data = 4 * (curr_node.data / 5) - 1 / 5
            else:
                newNode.data = 0
            
            newNode.checkmated = True
        else:
            legalMoves = board.find_moves(newBoard,whosMoveReverse,EnPassant)
            for move2 in legalMoves:
                newBoard2 = copy.deepcopy(newBoard)

                for moves1 in move2:
                    newBoard2, EnPassant = board.make_move(newBoard2[moves1[0][0]][moves1[0][1]],moves1[1],newBoard2)
        
                newNode2 = Node.Node(newBoard2, whosMove, move2)
                newNode.children.append(newNode2)
                newNode2.parent = newNode

                winlet = board.checkmate(newBoard2,whosMove)
                if winlet != False:
                    if winlet == whosMove:
                        newNode2.data = 1
                        newNode.data += 1 / len(legalMoves)
                        curr_node.data = 4 * (curr_node.data / 5) + newNode.data / 5
                    elif winlet == whosMoveReverse:
                        newNode2.data = -1
                        newNode.data += -1 / len(legalMoves)
                        curr_node.data = 4 * (curr_node.data / 5) - newNode.data / 5
                    else:
                        newNode2.data = 0
                    
                    newNode2.checkmated = True
                    newNode2 = prevNode
                else:
                    prevNode = newNode2

    if newNode2 != None:
        nodes = []
        for child in curr_node.children:
            if child.children != []:
                for gchild in child.children:
                    if gchild.data == 0:
                        nodes.append(gchild)
        
        newNode2 = random.choice(nodes)


    return newNode2

def backpropogation(curr_node: Node, reward):
    addReward = reward
    while(curr_node != None):
        curr_node.data = 4 * (curr_node.data / 5) + addReward / 5
        curr_node.sims += 1
        curr_node = curr_node.parent

        if (curr_node != None):
            checkmated = True
            for child in curr_node.children:
                if not child.checkmated:
                    checkmated = False
                    break

            if checkmated:
                curr_node.checkmated = True


    return curr_node

def firstBlackExpansion(curr_node, EnPassant):
    newNode = None
    legalMoves = board.find_moves(curr_node.chessBoard,"w",EnPassant)

    for move in legalMoves:
        newBoard = copy.deepcopy(curr_node.chessBoard)

        for moves1 in move:
            newBoard, EnPassant = board.make_move(newBoard[moves1[0][0]][moves1[0][1]],moves1[1],newBoard)
        
        newNode = Node.Node(newBoard, "b", move)
        curr_node.children.append(newNode)
        newNode.parent = curr_node
        