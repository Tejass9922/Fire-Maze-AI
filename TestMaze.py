#!/usr/bin/env python
# coding: utf-8

#todo 
#     Strategy 2:  BFS 
#     Strategy 3:  
#     Make loops for BFS / DFS 
#     Test highest dim for BFS / DFS at p = .3
#     Plot for DFS / BFS avg success


import numpy as np
import pandas as pd
import random
from collections import deque
from copy import copy, deepcopy
from queue import PriorityQueue
from dataclasses import dataclass, field
import matplotlib.pyplot as plt

# queue node used in BFS
class Node:
    # (x, y) represents coordinates of a cell in matrix
    # maintain a parent node for printing path
    def __init__(self, x, y, parent):
        self.x = x
        self.y = y
        self.parent = parent
    def __eq__(self,other):
        return (self.x == other.x) and (self.y == other.y)
    def __repr__(self):
        return str((self.x, self.y))
class ENode:
    # (x, y) represents coordinates of a cell in matrix
    # maintain a parent node for printing path
    def __init__(self, x, y, parent,distance,priority):
        self.x = x
        self.y = y
        self.parent = parent
        self.distance = distance
        self.priority = priority
    def __lt__(self, other):
        return self.priority < other.priority
    def __eq__(self,other):
        return (self.x == other.x) and (self.y == other.y)
    def __repr__(self):
        return str((self.x, self.y))



row = [-1, 0, 0, 1]
col = [0, -1, 1, 0]
q = .2

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
   
    #print(Matrix)
    

'''
testMatrix = createMatrix(3500, .30)
#dfsTestMatrix = deepcopy(testMatrix)
#astar_testMatrix = deepcopy(testMatrix)

bfsPathMatrix = deepcopy(testMatrix)

dfsPathMatrix = deepcopy(testMatrix)
astarPathMatrix = deepcopy(testMatrix)
strat1Matrix = deepcopy(testMatrix)
strat2Matrix = deepcopy(testMatrix)
'''
#printMatrix(testMatrix)

bfs_nodes_explored = []
a_star_avg = []







def onFire(x,y,grid):
    k = 0
    for i in range(4):
        xp = x  + row[i]
        yp = y  + col[i]
        if (0 <= xp < len(grid)) and (0<= yp < len(grid)):
            if grid[xp][yp] == '!':
                k = k + 1
    return k

def advance_fire(curr_matrix,q):
    N = len(curr_matrix)
    new_grid = deepcopy(curr_matrix)
    for i in range(len(curr_matrix)):
        for j in range(len(curr_matrix[0])):
            if (curr_matrix[i][j] != '!' and curr_matrix[i][j] != '_'):
                k = onFire(i,j,curr_matrix) 
                prob = 1 - pow((1-q),k)
                if random.random() <= prob:
                    new_grid[i][j] = '!'
                    

    new_grid[0][0] = 'S'
    new_grid[N-1][N-1] = 'G'
    return new_grid

def strategy1(path, matrix):
    
    for curr in path:
        x = curr[0]
        y = curr[1]
        if matrix[x][y] == '!':
            print("Path Failed, Maze burned")
            return matrix
        else:
            matrix[x][y] = 'X'
            matrix = advance_fire(matrix)
        counter = counter + 1
    
    print("Successfully exited the maze")
    
    return matrix


def strat1_graph(path, matrix,q):
    for curr in path:
        x = curr[0]
        y = curr[1]
        if matrix[x][y] == '!':
            #print("Path Failed, Maze burned")
            return False
        else:
            matrix[x][y] = 'X'
            matrix = advance_fire(matrix,q)
        #counter = counter + 1
    
    #print("Successfully exited the maze")
    
    return True

def strat2_graph(path, matrix,q):
    total_path = set()
    ordered_path = []
    iterator_index = 0
    while (len(path) > 0):
        curr = path[0]
        x = curr[0]
        y = curr[1]
        matrix[x][y] = 'X'
        total_path.add((y,x))
        ordered_path.append((y,x))
        coord2x = path[len(path)-1][0]
        coord2y = path[len(path)-1][1]
        coord2 = (coord2x,coord2y)
        nodeTemp = BFS(curr,coord2,matrix)
        path = getPathArray(nodeTemp)
        if len(path) == 0:
            '''
            print("Paths failed, Maze Burned")
            print("Attempted Path: ")
            for i in ordered_path:
                print(i, end = ' ')
            print("")
            '''
            return False
        else:
            #remove first element in path array 
            
            path = path[1:]
            matrix = advance_fire(matrix,q)
           
    '''
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if matrix[i][j] =='!' and (i,j) in total_path:
                matrix[i][j] = 'B'
    print("Successfully exited maze: ")
    print("Path Taken: ")
    for i in ordered_path:
        print(i, end = ' ')

    print("")
    '''
    return True
def strat3_graph(path,matrix,q):
    start = path[0]
    total_path = set()
    ordered_path = []
    iterator_index = 0
    fireMap = buildFireMap(matrix)
    while (len(path) > 0):
        
        curr = path[0]
        
        x = curr[0]
        y = curr[1]
        check_future_iterations = False
        i = 0
        while (i < len(path)):
            t_curr = path[i]
            t_x = t_curr[0]
            t_y = t_curr[1]
            if fireMap[t_x][t_y] >=.2:
                check_future_iterations = True
                break
            i = i  + 1
        if (matrix[x][y] == '!' or check_future_iterations):
            coord2x = path[len(path)-1][0]
            coord2y = path[len(path)-1][1]
            coord2 = (coord2x,coord2y)
            coord1 = (ordered_path[-1][1],ordered_path[-1][0])
            nodeTemp = a_star_s3(coord1 if len(ordered_path) > 0 else start ,coord2,matrix,fireMap)
            path = getPathArray(nodeTemp)
        
        if len(path) == 0:
            return False

        else:
            new_curr = path[0]
            x = new_curr[0]
            y = new_curr[1]
            matrix[x][y] = 'X'
            total_path.add((y,x))
            ordered_path.append((y,x))
            path = path[1:] 

        matrix = advance_fire(matrix,q)
        fireMap = buildFireMap(matrix)

    
    return True

def strategy2(path,matrix,q):
    
    #Explore even if no path??? or return as soon as there is no path left ? 
    
    total_path = set()
    ordered_path = []
    iterator_index = 0
    while (len(path) > 0):
        curr = path[0]
        x = curr[0]
        y = curr[1]
        matrix[x][y] = 'X'
        total_path.add((y,x))
        ordered_path.append((y,x))
        coord2x = path[len(path)-1][0]
        coord2y = path[len(path)-1][1]
        coord2 = (coord2x,coord2y)
        nodeTemp = BFS(curr,coord2,matrix)
        path = getPathArray(nodeTemp)
       
        if len(path) == 0:
            print("Paths failed, Maze Burned")
            print("Attempted Path: ")
            for i in ordered_path:
                print(i, end = ' ')
            print("")
            return matrix
        else:
            #remove first element in path array 
            
            path = path[1:]
            matrix = advance_fire(matrix,q)
           

    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if matrix[i][j] =='!' and (j,i) in total_path:
                matrix[i][j] = 'B'
    print("Successfully exited maze: ")
    print("Path Taken: ")
    for i in ordered_path:
        print(i, end = ' ')

    print("")
    return matrix
def heu_s3(pointA,pointB,fireMap):
    point1 = np.array(pointA) 
    point2 = np.array(pointB) 
  
    # calculating Euclidean distance 
    # using linalg.norm() 
    heu = np.linalg.norm(point1 - point2)  + ((fireMap[pointA[0]][pointA[1]])*10)

    return heu

def a_star_s3(Coord1,Coord2,Matrix,fireMap):
    pq = PriorityQueue()
    start = ENode(Coord1[0],Coord1[1],None,0,0)
    counter = 0
    if (Matrix[Coord2[0]][Coord2[1]] == '!') or (Matrix[Coord1[0]][Coord1[1]] == '!') or (Matrix[Coord1[0]][Coord1[1]] == '_') or (Matrix[Coord1[0]][Coord1[1]] == '_'):
        return None

    pq.put(start)
    visited = set()
    while not  pq.empty():
        curr = pq.get()
       
        xPos = (curr.x)
        yPos = (curr.y)
       
        #Update Maze for Fire

        # 1. Visited 
        # 2. Barrier
            #Compare the value  in the given matrix location
        # 3. Boundries
            #Compare the x and y boundries
        if ( xPos < 0  or xPos >= len(Matrix) or yPos < 0 or yPos >= len(Matrix[0])  or ((curr.x,curr.y) in visited) or ((Matrix[curr.x][curr.y]) == '_') or Matrix[curr.x][curr.y] == '!' ):
            continue
        counter = counter + 1
        if (xPos == Coord2[0] and yPos == Coord2[1]):
            a_star_avg.append(counter)
            return curr
        
        visited.add((curr.x,curr.y))
        
        if (Matrix[Coord2[0]][Coord2[1]] == '!'):
            return None

        for i in range(4):
            x = xPos + row[i]
            y = yPos + col[i]
            if (x < 0 or x >= len(Matrix) or y < 0 or y >= len(Matrix[0])):
                continue
            hue = heu_s3((x,y),Coord2,fireMap)
            dist = (curr.distance if curr.parent else 0)  + 1
            cost = hue  + dist
            next = ENode(x,y,curr,dist,cost)

            inSet = next in pq.queue
            for open_node in pq.queue:
                if open_node == next and ():
                    if curr.distance + 1 < open_node.distance:
                        open_node.distance = curr.distance + 1
                        open_node.parent = curr
                        break
            else:
                pq.put(next)
                    
            
    return None

    '''
    for loop -> all of PQ


    '''
def strategy3(path,matrix,q):
    start = path[0]
    total_path = set()
    ordered_path = []
    iterator_index = 0
    fireMap = buildFireMap(matrix)
    while (len(path) > 0):
        
        curr = path[0]
        
        x = curr[0]
        y = curr[1]
        check_future_iterations = False
        i = 0
        while (i < len(path)):
            t_curr = path[i]
            t_x = t_curr[0]
            t_y = t_curr[1]
            if fireMap[t_x][t_y] >=.2:
                check_future_iterations = True
                break
            i = i  + 1
        if (matrix[x][y] == '!' or check_future_iterations):
            coord2x = path[len(path)-1][0]
            coord2y = path[len(path)-1][1]
            coord2 = (coord2x,coord2y)
            coord1 = (ordered_path[-1][1],ordered_path[-1][0])
            nodeTemp = a_star_s3(coord1 if len(ordered_path) > 0 else start ,coord2,matrix,fireMap)
            path = getPathArray(nodeTemp)
        
        if len(path) == 0:
            print("Paths failed, Maze Burned")
            print("Attempted Path: ")
            for i in ordered_path:
                print(i, end = ' ')
            print("")
            return matrix

        else:
            new_curr = path[0]
            x = new_curr[0]
            y = new_curr[1]
            matrix[x][y] = 'X'
            total_path.add((y,x))
            ordered_path.append((y,x))
            path = path[1:] 
            matrix = advance_fire(matrix,q)
            fireMap = buildFireMap(matrix)

        

    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if matrix[i][j] =='!' and (j,i) in total_path:
                matrix[i][j] = 'B'
    
    print("Successfully exited maze: ")
    print("Path Taken: ")
    for i in ordered_path:
        print(i, end = ' ')

    print("")
    return matrix
    

def buildFireMap(matrix):
    fireMap = [ [ 0 for i in range(len(matrix)) ] for j in range(len(matrix)) ]

    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if (matrix[i][j] != '!' and matrix[i][j] != '_'):
                k = onFire(i,j,matrix)
                sample = []
                for x in range(10):
                    prob = 1 - pow((1-q),k)
                    sample.append(prob)
                avg = float(sum(sample) / float(len(sample)))
                fireMap[i][j] = avg
            elif matrix[i][j] == '_':
                fireMap[i][j] = 0
            elif matrix[i][j] == '!':
                fireMap[i][j] = 1

    return np.array(fireMap)


def getPathArray(node):
    stack = []
    while node:
        stack.append((node.x,node.y))
        node = node.parent

    prime_path = []
    while stack:
        prime_path.append(stack.pop())

    return prime_path
    




dfsPath = []
def DFSsearch(Coord1, Coord2, Matrix):
    stack = []
    start = Node(Coord1[0],Coord1[1],None)
    if (Matrix[Coord2[0]][Coord2[1]] == '!') or (Matrix[Coord1[0]][Coord1[1]] == '!') or (Matrix[Coord1[0]][Coord1[1]] == '_') or (Matrix[Coord1[0]][Coord1[1]] == '_'):
        return False
 
    
    stack.append(start)
    
    while stack:
        
        curr = stack.pop()

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

        #Also update fire??
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
            stack.append(next)

        '''
        stack.append((xPos + 1,yPos))
        stack.append((xPos -1, yPos))
        stack.append((xPos, yPos + 1))
        stack.append((xPos, yPos - 1))
        '''
    return None

def BFS(Coord1, Coord2, Matrix):
    q = deque()
    visited = set()
    
    if (Matrix[Coord2[0]][Coord2[1]] == '!') or (Matrix[Coord1[0]][Coord1[1]] == '!') or (Matrix[Coord1[0]][Coord1[1]] == '_') or (Matrix[Coord2[0]][Coord2[1]] == '_'):
        return None
 
    start = Node(Coord1[0],Coord1[1], None)
    q.append(start)
    counter = 0
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
        
        if ( xPos < 0  or xPos >= len(Matrix) or yPos < 0 or yPos >= len(Matrix[0])  or ((curr.x,curr.y) in visited) or ((Matrix[curr.x][curr.y]) == '_') or Matrix[curr.x][curr.y] == '!' ):
            continue
        
        counter = counter  + 1

        if (xPos == Coord2[0] and yPos == Coord2[1]):
            bfs_nodes_explored.append(counter)
            return curr
        
        visited.add((curr.x,curr.y))
        
        
        
    
    
        #print(str(xPos) + "\t" + str(yPos))
            
        
        #Check up
        #Check Down 
        #Check Right 
        #Check Left
       
        for i in range(4):
            x = xPos + row[i]
            y = yPos + col[i]
            next = Node(x,y,curr)
            if next not in q:
                q.append(next)

        '''
        queue.append((xPos + 1,yPos))
        queue.append((xPos -1, yPos))
        queue.append((xPos, yPos + 1))
        queue.append((xPos, yPos - 1))
        '''
    bfs_nodes_explored.append(counter)
    return None
    
def heuristic(pointA, pointB):
    point1 = np.array(pointA) 
    point2 = np.array(pointB) 
  
    # calculating Euclidean distance 
    # using linalg.norm() 
    hue = np.linalg.norm(point1 - point2)
    return hue

def a_star(Coord1, Coord2, Matrix):
    visited = set()
    pq = PriorityQueue()
    start = ENode(Coord1[0],Coord1[1],None,0,0)
    counter = 0
    if (Matrix[Coord2[0]][Coord2[1]] == '!') or (Matrix[Coord1[0]][Coord1[1]] == '!') or (Matrix[Coord1[0]][Coord1[1]] == '_') or (Matrix[Coord1[0]][Coord1[1]] == '_'):
        return None

    pq.put(start)

    while not pq.empty():
        curr = pq.get()
       
        xPos = (curr.x)
        yPos = (curr.y)
       
        #Update Maze for Fire

        # 1. Visited 
        # 2. Barrier
            #Compare the value  in the given matrix location
        # 3. Boundries
            #Compare the x and y boundries
        if ( xPos < 0  or xPos >= len(Matrix) or yPos < 0 or yPos >= len(Matrix[0])  or ((curr.x,curr.y) in visited) or ((Matrix[curr.x][curr.y]) == '_') or Matrix[curr.x][curr.y] == '!' ):
            continue
        counter = counter + 1
        if (xPos == Coord2[0] and yPos == Coord2[1]):
            a_star_avg.append(counter)
            return curr
        
        visited.add((curr.x,curr.y))
        
        if (Matrix[Coord2[0]][Coord2[1]] == '!'):
            return None

        for i in range(4):
            x = xPos + row[i]
            y = yPos + col[i]
           
            hue = heuristic((x,y),Coord2)
            dist = (curr.parent.distance if curr.parent else 0)  + 1
            cost = hue  + dist
            next = ENode(x,y,curr,dist,cost)
            '''
            if not pq.empty():
                print(pq.queue[0].priority)
            '''
            for open_node in pq.queue:
                print(next,end = ' ')
                print(open_node)
                if next == open_node and next.distance >= open_node.distance:
                  
                    break
            else:
               pq.put(next)
            
            
            
    a_star_avg.append(counter)
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

def startFire(matrix):
    randX = random.randint(0,len(matrix)-1)
    randY = random.randint(0,len(matrix)-1)
    while (matrix[randX][randY] != '0'):
        randX = random.randint(0,len(matrix)-1)
        randY = random.randint(0,len(matrix)-1)

    matrix[randX][randY] = '!'
    return matrix


#print(DFSsearch(a, b, testMatrix))
#print(dfsTestMatrix)
testMatrix = createMatrix(10,0)
'''
arr = [['S','0','0','_','0','_','0','0','0','0'],
 ['0','0','0','0','0','_','0','_','_','0'],
 ['_','!','_','0','0','0','0','0','_','0'],
 ['!','!','!','0','_','0','0','0','0','_'],
 ['!','!','!','0','_','_','0','0','_','0'],
 ['0','!','!','0','0','0','0','0','0','0'],
 ['0','0','0','0','_','0','0','0','0','0'],
 ['0','0','0','0','_','_','0','_','0','_'],
 ['0','0','0','0','0','0','0','0','0','0'],
 ['0','0','_','0','_','0','0','0','0','G']]
arr = np.array(arr)
'''
'''
a = (0,0)
b = (9,9)

test123 = deepcopy(testMatrix)
fireMap = buildFireMap(test123)
s3_astar = a_star_s3(a,b,test123,fireMap)

stack = []
if s3_astar:
     while s3_astar:
        stack.append((s3_astar.x,s3_astar.y))
        s3_astar = s3_astar.parent

     prime_path = stack[::-1] 
     print(prime_path)
else:
    print("no path")
'''  

'''


#dfsTemp = DFSsearch(a,b,dfsTestMatrix)
'''
'''
print("Testing Strategy 3 ------------------------------------|")

strat3matrix = deepcopy(testMatrix)
startFire(strat3matrix)
fireMap = buildFireMap(strat3matrix)
s3_astar = a_star_s3(a,b,strat3matrix,fireMap)
stack = []
x = s3_astar is not None 
if x:
    while s3_astar:
        stack.append((s3_astar.x,s3_astar.y))
        s3_astar = s3_astar.parent

    prime_path = stack[::-1] 
    print(strategy3(prime_path,strat3matrix,q))
else:
    print("No path")

'''
'''
astarPathMatrix = deepcopy(testMatrix)

astar_testMatrix = deepcopy(testMatrix)
astarTemp = a_star(a,b,astar_testMatrix)
'''
'''
testFire = deepcopy(testMatrix)
startFire(testFire)
for i in range(10):
    testFire = advance_fire(testFire,.3)
print(testFire)
print(buildFireMap(testFire))
'''
'''

'''
'''
if dfsTemp:
  
    print("DFS Path: ")
    l = getPath(dfsTemp,dfsPathMatrix)
    print(l)
else:
    print("no path")
'''
'''
if bfsTemp: 
   
    print("BFS Path: ")
    l = getPath(bfsTemp,bfsPathMatrix)
    print(l)

else:
    print("no path")
'''
'''
if astarTemp:
    print("A* Path: ")
    l = getPath(astarTemp,astarPathMatrix)
    print(l)
else:
    print("no path")
print(heuristic(a,b))
'''
'''
print("Trying Strategy 1---------------------------------------|")
stack = []
x = bfsTemp is not None 
if x:
    
    while bfsTemp:
        stack.append((bfsTemp.x,bfsTemp.y))
        bfsTemp = bfsTemp.parent

    prime_path = []
    while stack:
        prime_path.append(stack.pop())

    strat1Matrix = startFire(strat1Matrix)
    print(strategy1(prime_path,strat1Matrix))    
'''
'''
print("Trying Strategy 2--------------------------------------|")
stack = []
x = bfsTemp is not None 
if x:
    
    while bfsTemp:
        stack.append((bfsTemp.x,bfsTemp.y))
        bfsTemp = bfsTemp.parent

    prime_path = []
    while stack:
        prime_path.append(stack.pop())

    strat2Matrix = startFire(strat2Matrix)
    print(strategy2(prime_path,strat2Matrix,.2))

 
'''
'''
#-- DFS success rate vs obsticle density (p)
obsticle_density= np.linspace(.1,1,10)
dfs_success_counter = 0
success_tracker = []
for p in obsticle_density:
    for i  in  range(100):
        loop_matrix = createMatrix(10,p)
        N = len(loop_matrix) - 1
        a = (0,0)
        b = (N,N)
        dfsNode = DFSsearch(a,b,loop_matrix)
        if dfsNode:
            dfs_success_counter += 1

    success_tracker.append(dfs_success_counter / float(100))
    dfs_success_counter = 0

print(obsticle_density)
print(success_tracker)

plt.plot(obsticle_density,success_tracker)

#--- BFS - A star nodes explored vs obsticle density (p)---

diff = []
for p in obsticle_density:
    for i  in  range(100):
        loop_matrix = createMatrix(10,p)
        N = len(loop_matrix) - 1
        a = (0,0)
        b = (N,N)
        bfsNode = BFS(a,b,loop_matrix)
        a_star_node = a_star(a,b,loop_matrix)
        
    bfs_avg_nodes_explored = float((sum(bfs_nodes_explored) / len(bfs_nodes_explored)))
    a_star_avg_nodes = (sum(a_star_avg) / len(a_star_avg))
    diff.append(bfs_avg_nodes_explored - a_star_avg_nodes)
    bfs_nodes_explored = []
    a_star_avg = []
    diff = [round(num, 2) for num in diff]

'''

#--- Strategy 1  and  Strategy 2 success rate vs. fire intensity (q) with stable obstacle density (p = .3)---
fire_rate = np.linspace(.1,1,10)
success_strat1 = []
success_strat2 = []
success_strat3 = []
s1_count = 0
s2_count = 0
s3_count = 0
for qf in fire_rate:
    for j in range(100):
        strat1 = createMatrix(9,.3)
        strat2 = deepcopy(strat1)
        strat3 = deepcopy(strat1)
        N = len(strat1) - 1
        a = (0,0)
        b = (N,N)
        bfsNode = BFS(a,b,strat1)
        stack = []
        if (bfsNode is not None):
            while bfsNode:
                stack.append((bfsNode.x,bfsNode.y))
                bfsNode = bfsNode.parent

            prime_path = []
            while stack:
                prime_path.append(stack.pop())

            strat1Matrix = startFire(strat1)
            strat2Matrix = deepcopy(strat1Matrix)
            strat3Matrix = deepcopy(strat1Matrix)
            strat_1_result = strat1_graph(prime_path,strat1Matrix,qf)
            strat_2_result = strat2_graph(prime_path,strat2Matrix,qf)  
            
            if strat_1_result:
                s1_count += 1
            
            if strat_2_result:
                s2_count += 1

            fireMap = buildFireMap(strat1)
            astar_node = a_star_s3(a,b,strat1,fireMap)  
            stack = []
            x = astar_node is not None 
            if x:
                while astar_node:
                    stack.append((astar_node.x,astar_node.y))
                    astar_node = astar_node.parent

                prime_path = stack[::-1] 
                strat3_result = strat3_graph(prime_path,strat3Matrix,qf)

                if strat3_result:
                    s3_count += 1


    success_strat1.append(s1_count)
    success_strat2.append(s2_count)
    success_strat3.append(s3_count)
    s1_count = 0
    s2_count = 0
    s3_count = 0


print("")
print(success_strat1)
print(success_strat2)
print(success_strat3)
print(fire_rate)

