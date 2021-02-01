#!/usr/bin/env python
# coding: utf-8




import numpy as np
import pandas as pd
import random
from collections import deque
from copy import copy, deepcopy


# queue node used in BFS
class Node:
    # (x, y) represents coordinates of a cell in matrix
    # maintain a parent node for printing path
    def __init__(self, x, y, parent):
        self.x = x
        self.y = y
        self.parent = parent
 
    def __repr__(self):
        return str((self.x, self.y))

row = [-1, 0, 0, 1]
col = [0, -1, 1, 0]

def createMatrix(Dim, p):
    Matrix = [ [ 0 for i in range(Dim) ] for j in range(Dim) ]

    for i in range(Dim):
        for j in range(Dim):
            Matrix[i][j] = '0'
    for i in range(Dim):
        for j in range(Dim):
            if ((i != 0) | (j != 0)) and ((i != (Dim-1)) | (j != (Dim-1))):
                if random.random() < p:
                    Matrix[i][j]= '_'
                    
    Matrix[0][0] = 'S'
    Matrix[Dim-1][Dim-1] = 'G'
        
    return np.array(Matrix)

def printMatrix(Matrix):
    print('0 = unexpored')
    print('_ = Barrier')
    print('! = Fire')
    print('1 = Explored')
   
    print(Matrix)
    


testMatrix = createMatrix(10, .44)
bfsPathMatrix = deepcopy(testMatrix)
printMatrix(testMatrix)
'''
for i in range(len(testMatrix)):
    for j in range(len(testMatrix[0])):
        print(str(i) + "," + str(j), end = ' ')
        
    print("")
'''

#todo: Search Algos
#Strategy
#Fire (BFS Algo, Fringe (Queue)




a = (0,0)
b = (8,7)

'''def CoordinateCheck(Matrix, coordinate):
    xPos = coordinate[0]
    yPos = coordiante[1]
    MatrixDim = len(Matrix)
    
    if ()
    
'''
def advance_fire(curr_matrix):
    new_matrix = deepcopy(curr_matrix)

dfsPath = []
def DFSsearch(Coord1, Coord2, Matrix):
    stack = []
   
    if (Matrix[Coord2[0]][Coord2[1]] == '!') or (Matrix[Coord1[0]][Coord1[1]] == '!') or (Matrix[Coord1[0]][Coord1[1]] == '_') or (Matrix[Coord1[0]][Coord1[1]] == '_'):
        return False
 
    
    stack.append(Coord1)
    
    while(len(stack)):
        
        s = stack.pop()
      
        xPos = int(s[0])
        yPos = int(s[1])
       
        #Update Maze for Fire

        # 1. Visited 
        # 2. Barrier
            #Compare the value  in the given matrix location
        # 3. Boundries
            #Compare the x and y boundries
        if ( xPos < 0  or xPos >= len(Matrix) or yPos < 0 or yPos >= len(Matrix[0])  or ((Matrix[s[0]][s[1]]) == '1') or ((Matrix[s[0]][s[1]]) == '_')):
            continue

        #Also update fire??
        Matrix[xPos][yPos] = '1'
        
        if (Matrix[Coord2[0]][Coord2[1]] == '!'):
            return False
    
    
        if (xPos == Coord2[0] and yPos == Coord2[1]):
            return True
        
        #print(str(xPos) + "\t" + str(yPos))
            
        
        #Check up
        #Check Down 
        #Check Right 
        #Check Left
       
        stack.append((xPos + 1,yPos))
        stack.append((xPos -1, yPos))
        stack.append((xPos, yPos + 1))
        stack.append((xPos, yPos - 1))
        
    return False

def BFS(Coord1, Coord2, Matrix):
    q = deque()
    

    if (Matrix[Coord2[0]][Coord2[1]] == '!') or (Matrix[Coord1[0]][Coord1[1]] == '!') or (Matrix[Coord1[0]][Coord1[1]] == '_') or (Matrix[Coord1[0]][Coord1[1]] == '_'):
        return None
 
    start = Node(Coord1[0],Coord1[1], None)
    q.append(start)
    
    while q:
        
        curr = q.popleft()
      
        xPos = int(curr.x)
        yPos = int(curr.y)
       
        #Update Maze for Fire

        # 1. Visited 
        # 2. Barrier
            #Compare the value  in the given matrix location
        # 3. Boundries
            #Compare the x and y boundries
        if ( xPos < 0  or xPos >= len(Matrix) or yPos < 0 or yPos >= len(Matrix[0])  or ((Matrix[curr.x][curr.y]) == '1') or ((Matrix[curr.x][curr.y]) == '_')):
            continue

        
        Matrix[xPos][yPos] = '1'
        
        if (Matrix[Coord2[0]][Coord2[1]] == '!'):
            return None
    
    
        if (xPos == Coord2[0] and yPos == Coord2[1]):
            return curr
        
        #print(str(xPos) + "\t" + str(yPos))
            
        
        #Check up
        #Check Down 
        #Check Right 
        #Check Left
       
        for i in range(4):
            x = xPos + row[i]
            y = yPos + col[i]
            next = Node(x,y,curr)
            q.append(next)

        '''
        queue.append((xPos + 1,yPos))
        queue.append((xPos -1, yPos))
        queue.append((xPos, yPos + 1))
        queue.append((xPos, yPos - 1))
        '''
    return None
    
def getPath(node,matrix):
    stack = []
    temp = matrix
    while node:
        #print("("+str(node.y) + " , "  + str(node.x)  + ")", end = ' ')
        stack.append((node.y,node.x))
        temp[node.x][node.y] = 'X'
        node = node.parent

   
    while stack:
        print(stack.pop(), end = ' ') 
    print("")
    return temp
#print(DFSsearch(a, b, testMatrix))

temp = BFS(a,b,testMatrix)
if temp: 
    print(testMatrix)
    print("Path: ")
    l = getPath(temp,bfsPathMatrix)
    print(l)
else:
    print("no path")




