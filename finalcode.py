import random
import time
import statistics
import numpy as np
import matplotlib.pyplot as plt
import math

#Grid Class
class Grid:

    #initializes Grid
    def __init__(self, size, listLength, traits, rows, cols):
        self.width = cols
        self.height = rows
        self.length = listLength
        self.traits = traits
        self.size = size
        W = self.width
        H = self.height
        L = self.length
        self.data = [[['0' for leng in range(L) ] for widt in range(W)] for row in range(H) ]
        self.neighbors = [[[] for widt in range(W)] for row in range(H) ]
        self.counts = [[0 for widt in range(W)] for row in range(H) ]

    #allows grid to be printed
    def __repr__(self):
        H = self.height
        W = self.width
        L = self.length
        s = ''   # the string to return
        for row in range(0,H):          
            for col in range(0,W):
                s += '|'
                for leng in range(0,L):
                    s+= self.data[row][col][leng]
            s += '\n'
        s+= '\n'
        for row in range(0,H):          
            for col in range(0,W):
                s += '|'
                
                s+= str(self.counts[row][col])
            s += '\n'
        return s
    
    def makeNeighbors(self):
        H = self.height
        W = self.width
        S = self.size
        for row in range(0,H):
            for col in range(0,W):
                self.neighbors[row][col] = self.neighborhood(row,col,S)

    #creates a random board             
    def createRandom(self):
        H = self.height
        W = self.width
        L = self.length
        T = self.traits
        S = self.size
        K = []
        for row in range(0,H):
            for col in range(0,W):
                self.neighbors[row][col] = self.neighborhood(row,col,S)
                for leng in range(0,L):
                    for i in range(T):
                        K.append(str(i))
                        self.data[row][col][leng]=random.choice(K)

    #creates the neighborhoods of each agent
    def neighborhood(self,row,col,size):
        H=self.height
        neighborhood=[]
        if size == 4:
            neighborhood.append([row+1,col])
            neighborhood.append([row-1,col])
            neighborhood.append([row,col+1])
            neighborhood.append([row,col-1])
            
        elif size == 8:
            neighborhood.append([row+1,col])
            neighborhood.append([row-1,col])
            neighborhood.append([row,col+1])
            neighborhood.append([row,col-1])
            neighborhood.append([row+1,col+1])
            neighborhood.append([row-1,col+1])
            neighborhood.append([row+1,col-1])
            neighborhood.append([row-1,col-1])
            
        elif size == 12:
            neighborhood.append([row+1,col])
            neighborhood.append([row-1,col])
            neighborhood.append([row,col+1])
            neighborhood.append([row,col-1])
            neighborhood.append([row+1,col+1])
            neighborhood.append([row-1,col+1])
            neighborhood.append([row+1,col-1])
            neighborhood.append([row-1,col-1])
            neighborhood.append([row+2,col])
            neighborhood.append([row-2,col])
            neighborhood.append([row,col+2])
            neighborhood.append([row,col-2])
        for j in range(2):
            neighborhood = [i for i in neighborhood if i[j]<H and i[j]>=0]
        return neighborhood
        

    #counts the number of regions
    #counts regions of the same features seperately if not connected
    def countRegion(self):
        R=[]
        H = self.height
        W = self.width
        for row in range(0,H):
            for col in range(0,W):
                if self.data[row][col] not in R:
                    R.append(self.data[row][col])
        return len(R)

    #counts the number of regions
    #counts disconnected regions of same features as one
    def countRegion2(self):
        arr = self.data
        x = self.height #height
        y = self.width #width 
        regions = [[None for i in range(y)] for i in range(x)]
        label = 0
        queue = []
        def check_neighbor(i, j, v):
            if not regions[i][j] and arr[i][j] == v:
                regions[i][j] = label
                queue.insert(0, (i, j))
        for i in range(x):
            for j in range(y):
                if regions[i][j]: continue
                label += 1 
                regions[i][j] = label
                queue = [(i, j)]
                v = arr[i][j]
                while queue:
                    (X, Y) = queue.pop()
                    if X > 0:
                        check_neighbor(X-1, Y, v)
                    if X < x-1:
                        check_neighbor(X+1, Y, v)
                    if Y > 0:
                        check_neighbor(X, Y-1, v)
                    if Y < y-1:
                        check_neighbor(X, Y+1, v)   
        return label 

    #Calculates Walker's Disorder
    def disorder(self):
        W = self.width
        x = self.height
        y = self.length
        maxConnections = 2*W*(W-1)
        arr = self.data
        regions = [[None for i in range(y)] for i in range(x)]
        label = 0
        queue = []
        def check_neighbor(i, j, v):
            if not regions[i][j] and arr[i][j] == v:
                regions[i][j] = label
                queue.insert(0, (i, j))
        for i in range(x):
            for j in range(y):
                if regions[i][j]: continue
                label += 1                
                regions[i][j] = label
                queue = [(i, j)]
                v = arr[i][j]
                while queue:
                    (X, Y) = queue.pop()
                    if X > 0:
                        check_neighbor(X-1, Y, v)
                    if X < x-1:
                        check_neighbor(X+1, Y, v)
                    if Y > 0:
                        check_neighbor(X, Y-1, v)
                    if Y < y-1:
                        check_neighbor(X, Y+1, v)
        stuff=[[] for i in range(label)]
        for i in range(x):
            for j in range(y):
                stuff[regions[i][j]-1].append([i,j])
        connections=[0 for i in range(label)]
        for i in range(label):
            for j in stuff[i]:
                hood = self.neighbors[j[0]][j[1]]
                for n in hood:
                    if self.data[j[0]][j[1]] == self.data[n[0]][n[1]]:
                        connections[i] += 1     
        connections = [x/2 for x in connections]
        regionSizes=[0 for i in range(label)]
        for i in range(label):
            regionSizes[i] = len(stuff[i])
        N = x**2
        regionSizes = [x/N for x in regionSizes]
        numerator = 0
        for i in range(label):
            numerator = numerator + (regionSizes[i]*connections[i])
        disorder = numerator/maxConnections
        disorder = 1-disorder
        return disorder

    #calculates Landsberg Disorder               
    def landsbergDisorder(self):
        x = self.width
        y = self.height
        arr = self.data
        regions = [[None for i in range(y)] for i in range(x)]
        label = 0
        queue = []
        def check_neighbor(i, j, v):
            if not regions[i][j] and arr[i][j] == v:
                regions[i][j] = label
                queue.insert(0, (i, j))
        for i in range(x):
            for j in range(y):
                if regions[i][j]: continue
                label += 1                 
                regions[i][j] = label
                queue = [(i, j)]
                v = arr[i][j]
                while queue:
                    (X, Y) = queue.pop()
                    if X > 0:
                        check_neighbor(X-1, Y, v)
                    if X < x-1:
                        check_neighbor(X+1, Y, v)
                    if Y > 0:
                        check_neighbor(X, Y-1, v)
                    if Y < y-1:
                        check_neighbor(X, Y+1, v)
        stuff=[[] for i in range(label)]
        for i in range(x):
            for j in range(y):
                stuff[regions[i][j]-1].append([i,j])
        connections=[0 for i in range(label)]
        for i in range(label):
            for j in stuff[i]:
                hood = self.neighbors[j[0]][j[1]]
                for n in hood:
                    if self.data[j[0]][j[1]] == self.data[n[0]][n[1]]:
                        connections[i] += 1     
        connections = [x/2 for x in connections]
        regionSizes=[0 for i in range(label)]
        for i in range(label):
            regionSizes[i] = len(stuff[i])
        print(regionSizes)
        N = x**2
        regionSizes = [(x+1-math.sqrt(i)) for i in regionSizes]
        print(regionSizes)
        regionSizes = [math.log(i) for i in regionSizes]
        print(regionSizes)
        numerator = 0
        for i in range(label):
            numerator = numerator + regionSizes[i]
        denominator = label*math.log(x + 1 - math.sqrt(N/label))
        print(numerator)
        print(denominator)
        disorder = numerator/denominator        
        return disorder
   
    def disorder3(self):
        Matrix=self.data
        Sizes=self.regionSizes()
        H=len(Matrix)
        Sizes.sort()
        Sizes.insert(0,0)
        x=0
        N=H*H
        M=len(Sizes)
        for i in range(1,M):
            N+= -Sizes[i-1]
            p= math.sqrt(N)+1-math.sqrt(Sizes[i])
            x += 2*math.log(p)
        y=math.log(math.factorial(H*H))
        disorder = x/y
        return disorder
    
    def disorder5(self):
        Matrix=self.data
        H=len(Matrix)
        S=self.regionSizes()
        a=0
        for size in S:
            a+=(size/(H*H))*math.log(size/(H*H))
        LandsorderNew =-a
        return LandsorderNew

    def disorder6(self):
        Matrix=self.data
        H=len(Matrix)
        W=len(Matrix)
        LikeAgents=0
        NumConnections=0
        totalcomparisons=0
        for row in range(0,H-1):
            for col in range(0,W):
                NumConnections+=1
                totalcomparisons+=1
                if Matrix[row][col]== Matrix[row+1][col]:
                    LikeAgents+=1
        for row in range(0,H):
            for col in range(0,W-1):
                NumConnections+=1
                totalcomparisons+=1
                if Matrix[row][col] == Matrix[row][col+1]:
                    LikeAgents+=1
        unlikeAgents=totalcomparisons-LikeAgents
        disorder=unlikeAgents/NumConnections
        return disorder

    #counts the number of agents in each region
    def regionSizes(self):
        R=[]
        X=[]
        Y=[]
        H = self.height
        W = self.width
        for row in range(0,H):
            for col in range(0,W):
                if self.data[row][col] not in R:
                    R.append(self.data[row][col])
        sizes=[]
        for i in R:
            total=0
            for row in range(0,H):
                for col in range(0,W):
                    if self.data[row][col] == i:
                        total+=1
            sizes.append([i,total])
            X.append(i)
            Y.append(total)
        return sizes

    #counts number of regions
    #counts disconnected regions as the same
    def regionSizes2(self):
        x = self.height
        y = self.width
        arr = self.data
        regions = [[None for i in range(y)] for i in range(x)]
        label = 0
        queue = []
        def check_neighbor(i, j, v):
            if not regions[i][j] and arr[i][j] == v:
                regions[i][j] = label
                queue.insert(0, (i, j))
        for i in range(x):
            for j in range(y):
                if regions[i][j]: continue
                label += 1 
                regions[i][j] = label
                queue = [(i, j)]
                v = arr[i][j]
                while queue:
                    (X, Y) = queue.pop()
                    if X > 0:
                        check_neighbor(X-1, Y, v)
                    if X < x-1:
                        check_neighbor(X+1, Y, v)
                    if Y > 0:
                        check_neighbor(X, Y-1, v)
                    if Y < y-1:
                        check_neighbor(X, Y+1, v)
        sizes=[]
        for i in regions:
            total=0
            for row in range(0,x):
                for col in range(0,y):
                    if regions[row][col] == i:
                        total+=1
            sizes.append([i,total])
        return sizes
                    

    #Returns a list of different traits between two agents
    def differentTraits(self, row1, col1, neighbor):
        DifferentTraits = []
        L=self.length
        for i in range(0,L):
            if self.data[row1][col1][i] != self.data[neighbor[0]][neighbor[1]][i]:
                DifferentTraits.append(i)
        return DifferentTraits

    #Returns a random integer representing a row
    def getRandomRow(self):
        H = self.height
        row = random.randint(0,H-1)
        return row

    #Returns a random interger representing a column
    def getRandomCol(self):
        W = self.width
        col = random.randint(0,W-1)
        return col  

    #Returns true or false if the bond is active
    def activeBond(self, row1, col1, neighbor):
        L = self.length
        DifferentTraits=self.differentTraits(row1, col1, neighbor)
        if len(DifferentTraits)!=0 and len(DifferentTraits) != L:
            n=random.randint(0,L-1)
            if self.data[row1][col1][n] == self.data[neighbor[0]][neighbor[1]][n]:
                return True

    #Returns true or false if there is a possible mov ein the grid    
    def possibleMove(self):
        L = self.length
        H = self.height
        W = self.width
        for row in range(0,H-1):
            for col in range(0,W):
                if self.data[row][col]!=self.data[row+1][col]:
                    for n in range(0,L):
                        if self.data[row][col][n]==self.data[row+1][col][n]:
                            return True
        for row in range(0,H):
            for col in range(0,W-1):
                if self.data[row][col]!=self.data[row][col+1]:
                    for n in range(0,L):
                        if self.data[row][col][n]==self.data[row][col+1][n]:
                            return True
        return False

    #Returns an integer that is the number of features shared
    def overlap(self, row, col, neighbors, feature):
        overlap=0
        for i in neighbors:
            
            if self.data[row][col][feature]==self.data[i[0]][i[1]][feature]:
                overlap+=1
        return overlap

    #Goes through the process of changing a neighbor   
    def neighborChange(self, row, col):
        neighborhood=self.neighbors[row][col]
        #Random Neighbor Chosen
        neighbor = random.choice(neighborhood)    
        #Calculate probability of interaction
        if self.activeBond(row, col, neighbor)==True:
            #Interaction occurs so we proceed
            DifferentTraits = self.differentTraits(row, col, neighbor)          
            #Choose random unshared feature between i and j
            feature = random.choice(DifferentTraits)
            #calculate overlap between neighborhood and agent and selected neighbor
            agentOverlap = self.overlap(row, col, neighborhood, feature)
            neighborOverlap = self.overlap(neighbor[0], neighbor[1], neighborhood, feature)
            #add one as the agent compares itself to itself
            agentOverlap += 1
            #Calculate P2
            P2 = neighborOverlap/(agentOverlap+neighborOverlap)
            if random.random() < P2:
                self.data[row][col][feature] = self.data[neighbor[0]][neighbor[1]][feature]
                self.counts[row][col]+=1
                return 1
        return 0

    #Runs Code until stable
    def process(self):
        self.createRandom()
        print(self)
        self.makeNeighbors()
        start=time.time()
        i=0
        success=0     
        while self.possibleMove():
            row = self.getRandomRow()
            col = self.getRandomCol()
            success += self.neighborChange(row,col)
            success += self.neighborChange(row,col)
        end = time.time()
        totalTime= end-start
        print(self)
        print("The number of interaction is",i)
        print("The number of successful interaction is",success)
        if self.countRegion2()==1:
            print("There is a monocultural state!")
        else:
            print("There isn't a monoculture.")
            print("There are",self.countRegion2(),"cultural regions.")
        print("Took: ", totalTime, " seconds")
            
    #Returns a colored grid 
    def color(self,imageName):
        H = self.height
        W = self.width
        #Matrix=[[0 for col in range (W)] for row in range (H)]
        Matrix = np.zeros((H, W))
        stableRegions=[]
        for row in range (0,H):
            for col in range(0,W):
                if self.data[row][col] not in stableRegions:
                    stableRegions.append(self.data[row][col])
        for row in range (0,H):
            for col in range(0,W):
                for i in range (len(stableRegions)):
                    if self.data[row][col]==stableRegions[i]:
                        Matrix[row,col]=i+1     
        #print(Matrix)
        Matrix=np.flip(Matrix,0)
        xi = np.arange(0, W+1)
        yi = np.arange(0, H+1)
        X, Y = np.meshgrid(xi, yi)
        plt.pcolormesh(X, Y, Matrix)
        #plt.colorbar()
        plt.pcolormesh(Matrix, edgecolors="w")
        plt.axis('off')
        plt.title('Stable State of Axelrod Model')
        #plt.show()
        
        plt.savefig(imageName)
        plt.close()

    def color1(self, borderName):
        H = self.height #NumY
        W = self.width  #NumX
        L = self.length
        for row in range(0,H):
            for col in range(0,W-1):
                m=0
                for n in range(0,L):
                    if self.data[row][col][n]==self.data[row][col+1][n]:
                        m+=1
                if m==5:
                    x=[W-row,W-row-1]
                    y=[col+1,col+1]
                    plt.plot(y,x,zorder=1,c='white',linewidth=5)
                elif m==4:
                    x=[W-row,W-row-1]
                    y=[col+1,col+1]
                    plt.plot(y,x,zorder=1,c='lightblue',linewidth=5)
                elif m==3:
                    x=[W-row,W-row-1]
                    y=[col+1,col+1]
                    plt.plot(y,x,zorder=1,c='lightskyblue',linewidth=5)
                elif m==2:
                    x=[W-row,W-row-1]
                    y=[col+1,col+1]
                    plt.plot(y,x,zorder=1,c='deepskyblue',linewidth=5)
                elif m==1:
                    x=[W-row,W-row-1]
                    y=[col+1,col+1]
                    plt.plot(y,x,zorder=1,c='blue',linewidth=5)
                else:
                    x=[W-row,W-row-1]
                    y=[col+1,col+1]
                    plt.plot(y,x,zorder=1,c='midnightblue',linewidth=5)
        for row in range(0,H-1):
            for col in range(0,W):
                m=0
                for n in range(0,L):
                    if self.data[row][col][n]==self.data[row+1][col][n]:
                        m+=1    
                if m==5:
                    x=[W-row-1,W-row-1]
                    y=[col,col+1]
                    plt.plot(y,x,zorder=1,c='white',linewidth=5)
                elif m==4:
                    x=[W-row-1,W-row-1]
                    y=[col,col+1]
                    plt.plot(y,x,zorder=1,c='lightblue',linewidth=5)
                elif m==3:
                    x=[W-row-1,W-row-1]
                    y=[col,col+1]
                    plt.plot(y,x,zorder=1,c='lightskyblue',linewidth=5)
                elif m==2:
                    x=[W-row-1,W-row-1]
                    y=[col,col+1]
                    plt.plot(y,x,zorder=1,c='deepskyblue',linewidth=5)
                elif m==1:
                    x=[W-row-1,W-row-1]
                    y=[col,col+1]
                    plt.plot(y,x,zorder=1,c='blue',linewidth=5)
                else:
                    x=[W-row-1,W-row-1]
                    y=[col,col+1]
                    plt.plot(y,x,zorder=1,c='midnightblue',linewidth=5)              
        for row in range (H+1):
            for col in range (W+1):
                plt.scatter(row, col, c='k', marker="o",linewidths=0.02, zorder=2)
        plt.title('Steve Model 3.0')
        plt.axis('off')
        #plt.show()
        plt.savefig(borderName)


        



