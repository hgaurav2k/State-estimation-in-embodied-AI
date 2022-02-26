import random

class Grid(object):
    def __init__(self, layoutFile):
        with open(layoutFile) as f:
            self.grid = [line.strip() for line in f]
        self.height = len(self.grid)
        self.width = len(self.grid[0])
    def getHeight(self):
        return self.height
    def getWidth(self):
        return self.width
    def isWall(self, x, y):
        return self.grid[x][y] == '#'

class Robot(object):
    def __init__(self, grid, R):
        self.grid = grid
        self.R = R
        self.x = random.randint(0, grid.getHeight() - 1)
        self.y = random.randint(0, grid.getWidth() - 1)
        while(self.grid.isWall(self.x, self.y)):
            self.x = random.randint(0, grid.getHeight() - 1)
            self.y = random.randint(0, grid.getWidth() - 1)
    def move(self):
        options = []
        if self.x > 0 and not self.grid.isWall(self.x - 1, self.y):
            options.append((self.x - 1, self.y))
        if self.x < self.grid.getHeight() - 1 and not self.grid.isWall(self.x + 1, self.y):
            options.append((self.x + 1, self.y))
        if self.y > 0 and not self.grid.isWall(self.x, self.y - 1):
            options.append((self.x, self.y - 1))
        if self.y < self.grid.getWidth() - 1 and not self.grid.isWall(self.x, self.y + 1):
            options.append((self.x, self.y + 1))
        assert(len(options) > 0)
        self.x, self.y = random.choice(options)
    def getPosition(self):
        return (self.x, self.y)
    def getObservation(self):
        probabilities = []
        for direction in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
            x = self.x + direction[0]
            y = self.y + direction[1]
            while(not self.grid.isWall(x, y)):
                x += direction[0]
                y += direction[1]
            distance = abs(x - self.x) + abs(y - self.y)
            if(distance >= self.R):
                probability = 0
            else :
                probability = 1 - ((distance-1) / (self.R-1))
            probabilities.append(probability)
        observation = [random.choices([0, 1], [1 - p, p])[0] for p in probabilities]
        return tuple(observation)

class Model(object):
    def __init__(self, grid, R):
        self.grid = grid
        self.R = R
    def probabilityNextState(self, currState, nextState):
        if abs(nextState[0] - currState[0]) + abs(nextState[1] - currState[1]) != 1 or self.grid.isWall(nextState[0], nextState[1]):
            return 0
        neighbours = 0
        for direction in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
            x = currState[0] + direction[0]
            y = currState[1] + direction[1]
            if not self.grid.isWall(x, y):
                neighbours += 1
        assert(neighbours != 0)
        return 1 / neighbours
    def probabilityObservation(self, currState, observation):
        probability = 1
        for index, direction in enumerate([(-1, 0), (0, 1), (1, 0), (0, -1)]):
            x = currState[0] + direction[0]
            y = currState[1] + direction[1]
            while(not self.grid.isWall(x, y)):
                x += direction[0]
                y += direction[1]
            distance = abs(x - currState[0]) + abs(y - currState[1])
            if(distance >= self.R):
                alpha = 0
            else :
                alpha = 1 - ((distance-1) / (self.R-1))
            probability *= (alpha if observation[index] == 1 else (1 - alpha))
        return probability
    def updateBeliefWithTime(self, currBelief):
        nextBelief = [[0 for _ in range(self.grid.getWidth())] for _ in range(self.grid.getHeight())]
        for nextX in range(self.grid.getHeight()):
            for nextY in range(self.grid.getWidth()):
                if self.grid.isWall(nextX, nextY):
                    continue
                for currX in range(self.grid.getHeight()):
                    for currY in range(self.grid.getWidth()):
                        if self.grid.isWall(currX, currY):
                            continue
                        nextBelief[nextX][nextY] += self.probabilityNextState((currX, currY), (nextX, nextY)) * currBelief[currX][currY]
        return nextBelief
    def updateBeliefWithObservation(self, currBelief, observation):
        nextBelief = [[0 for _ in range(self.grid.getWidth())] for _ in range(self.grid.getHeight())]
        for x in range(self.grid.getHeight()):
            for y in range(self.grid.getWidth()):
                if self.grid.isWall(x, y):
                    continue
                nextBelief[x][y] = self.probabilityObservation((x, y), observation) * currBelief[x][y]
        total = sum(sum(row) for row in nextBelief)
        assert(total != 0)
        for x in range(self.grid.getHeight()):
            for y in range(self.grid.getWidth()):
                nextBelief[x][y] /= total
        return nextBelief
    def getMostLikelyPath(self, observations):
        T = len(observations)
        M = [[[0 for _ in range(self.grid.getWidth())] for _ in range(self.grid.getHeight())] for _ in range(T+1)]
        prev = [[[(-1, -1) for _ in range(self.grid.getWidth())] for _ in range(self.grid.getHeight())] for _ in range(T+1)]
        for x in range(self.grid.getHeight()):
            for y in range(self.grid.getWidth()):
                if self.grid.isWall(x, y):
                    continue
                M[0][x][y] = 1
        total = sum(sum(row) for row in M[0])
        assert(total != 0)
        for x in range(self.grid.getHeight()):
            for y in range(self.grid.getWidth()):
                M[0][x][y] /= total
        for t in range(1, T+1):
            for nextX in range(self.grid.getHeight()):
                for nextY in range(self.grid.getWidth()):
                    if self.grid.isWall(nextX, nextY):
                        continue
                    M[t][nextX][nextY] = 0
                    for x in range(self.grid.getHeight()):
                        for y in range(self.grid.getWidth()):
                            if self.grid.isWall(x, y):
                                continue
                            if self.probabilityNextState((x, y), (nextX, nextY)) * M[t-1][x][y] >= M[t][nextX][nextY]:
                                M[t][nextX][nextY] = self.probabilityNextState((x, y), (nextX, nextY)) * M[t-1][x][y]
                                prev[t][nextX][nextY] = (x, y)
                    M[t][nextX][nextY] *= self.probabilityObservation((nextX, nextY), observations[t-1])
            total = sum(sum(row) for row in M[t])
            assert(total != 0)
            for nextX in range(self.grid.getHeight()):
                for nextY in range(self.grid.getWidth()):
                    if self.grid.isWall(nextX, nextY):
                        continue
                    M[t][nextX][nextY] /= total
        
        maximumProbability = 0
        mostLikelyPath = []
        for x in range(self.grid.getHeight()):
            for y in range(self.grid.getWidth()):
                if self.grid.isWall(x, y):
                    continue
                if M[T][x][y] > maximumProbability:
                    maximumProbability = M[T][x][y]
                    mostLikelyPath = [(x, y)]
        for t in range(T, 1, -1):
            mostLikelyPath.append(prev[t][mostLikelyPath[-1][0]][mostLikelyPath[-1][1]])
        mostLikelyPath.reverse()
        return mostLikelyPath