from model import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import statistics
import math

R = 5
LAYOUT = 'layouts/closed.lay'
random.seed(4)

def q3a():
    grid = Grid(LAYOUT)
    robot = Robot(grid, R)
    model = Model(grid, R)

    print(grid.grid)
    return

    gridData = [[1 for _ in range(grid.getWidth())] for _ in range(grid.getHeight())]
    for x in range(grid.getHeight()):
        for y in range(grid.getWidth()):
            if grid.isWall(x, y):
                gridData[x][y] = 0
    
    belief = [[0 for _ in range(grid.getWidth())] for _ in range(grid.getHeight())]
    for x in range(grid.getHeight()):
        for y in range(grid.getWidth()):
            if not grid.isWall(x, y):
                belief[x][y] = 1
    total = sum(sum(row) for row in belief)
    for x in range(grid.getHeight()):
        for y in range(grid.getWidth()):
            belief[x][y] /= total
    
    path = []
    observations = []
    
    for t in range(1, 26):
        if t != 1:
            robot.move()
            belief = model.updateBeliefWithTime(belief)

        path.append(robot.getPosition())
        observation = robot.getObservation()
        observations.append(observation)
        # print(observation)
        belief = model.updateBeliefWithObservation(belief, observation)

        estimatedPosition = (0, 0)
        maxProbability = 0
        for x in range(grid.getHeight()):
            for y in range(grid.getWidth()):
                if belief[x][y] > maxProbability:
                    maxProbability = belief[x][y]
                    estimatedPosition = (x, y)

        plt.plot([robot.getPosition()[1]], [robot.getPosition()[0]], 'ro')
        plt.plot([estimatedPosition[1]], [estimatedPosition[0]], 'yo')
        plt.imshow(gridData, cmap='gray')
        plt.pause(0.3)
        plt.clf()
    
    plt.show()

    mostLikelyPath = model.getMostLikelyPath(observations)

    figure, axis = plt.subplots(2)

    X = [p[0] for p in path]
    Y = [p[1] for p in path]
    axis[0].plot(Y, X)
    axis[0].imshow(gridData, cmap='gray')

    X = [p[0] for p in mostLikelyPath]
    Y = [p[1] for p in mostLikelyPath]
    axis[1].plot(Y, X, 'y')
    axis[1].imshow(gridData, cmap='gray')

    plt.show()

    print('Path:', path)
    print('Most Likely Path:', mostLikelyPath)
    print('Observations:', observations)

def q3b():
    grid = Grid(LAYOUT)
    robot = Robot(grid, R)
    model = Model(grid, R)

    gridData = [[0 for _ in range(grid.getWidth())] for _ in range(grid.getHeight())]
    for x in range(grid.getHeight()):
        for y in range(grid.getWidth()):
            if grid.isWall(x, y):
                gridData[x][y] = -0.5
    
    belief = [[0.0 for _ in range(grid.getWidth())] for _ in range(grid.getHeight())]
    for x in range(grid.getHeight()):
        for y in range(grid.getWidth()):
            if not grid.isWall(x, y):
                belief[x][y] = 1
    total = sum(sum(row) for row in belief)
    for x in range(grid.getHeight()):
        for y in range(grid.getWidth()):
            belief[x][y] /= total
    
    for t in range(1, 26):
        if t != 1:
            robot.move()
            belief = model.updateBeliefWithTime(belief)

        observation = robot.getObservation()
        belief = model.updateBeliefWithObservation(belief, observation)

        for x in range(grid.getHeight()):
            for y in range(grid.getWidth()):
                if not grid.isWall(x, y):
                    gridData[x][y] = max(math.log(1 - belief[x][y]), -0.5)
        
        plt.plot([robot.getPosition()[1]], [robot.getPosition()[0]], 'ro')
        plt.imshow(gridData, cmap='gray')
        plt.pause(0.5)
        plt.clf()
    
    plt.show()

def q3c():
    estimatedPathErrors = []
    mostLikelyPathErrors = []
    for _ in range(50):
        print('Iteration:', _)
        grid = Grid(LAYOUT)
        robot = Robot(grid, R)
        model = Model(grid, R)

        belief = [[0 for _ in range(grid.getWidth())] for _ in range(grid.getHeight())]
        for x in range(grid.getHeight()):
            for y in range(grid.getWidth()):
                if not grid.isWall(x, y):
                    belief[x][y] = 1
        total = sum(sum(row) for row in belief)
        for x in range(grid.getHeight()):
            for y in range(grid.getWidth()):
                belief[x][y] /= total
        
        positionError = 0
        mostLikelyPathError = 0
        observations = []
        path = []
        for t in range(1, 26):
            if t != 1:
                robot.move()
                belief = model.updateBeliefWithTime(belief)
            observation = robot.getObservation()
            observations.append(observation)
            path.append(robot.getPosition())
            belief = model.updateBeliefWithObservation(belief, observation)

            estimatedPosition = (0, 0)
            maxProbability = 0
            for x in range(grid.getHeight()):
                for y in range(grid.getWidth()):
                    if belief[x][y] > maxProbability:
                        maxProbability = belief[x][y]
                        estimatedPosition = (x, y)
            
            positionError += abs(robot.getPosition()[0] - estimatedPosition[0]) + abs(robot.getPosition()[1] - estimatedPosition[1])
        mostLikelyPath = model.getMostLikelyPath(observations)
        for i in range(len(mostLikelyPath)):
            mostLikelyPathError += abs(path[i][0] - mostLikelyPath[i][0]) + abs(path[i][1] - mostLikelyPath[i][1])
        estimatedPathErrors.append(positionError)
        mostLikelyPathErrors.append(mostLikelyPathError)
    
    print('Position Estimation Error Mean:', statistics.mean(estimatedPathErrors))
    print('Position Estimation Error Std Dev:', statistics.stdev(estimatedPathErrors))
    print('Most Likely Path Error Mean:', statistics.mean(mostLikelyPathErrors))
    print('Most Likely Path Error Std Dev:', statistics.stdev(mostLikelyPathErrors))



if __name__ == '__main__':
    q3a()