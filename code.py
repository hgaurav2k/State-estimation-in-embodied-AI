import numpy as np
import matplotlib.pyplot as plt
from itertools import  permutations
import math
import random
import os,sys

## model for Q3 ##
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
        while (self.grid.isWall(self.x, self.y)):
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
        assert (len(options) > 0)
        self.x, self.y = random.choice(options)

    def getPosition(self):
        return (self.x, self.y)

    def getObservation(self):
        probabilities = []
        for direction in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
            x = self.x + direction[0]
            y = self.y + direction[1]
            while (not self.grid.isWall(x, y)):
                x += direction[0]
                y += direction[1]
            distance = abs(x - self.x) + abs(y - self.y)
            if (distance >= self.R):
                probability = 0
            else:
                probability = 1 - ((distance - 1) / (self.R - 1))
            probabilities.append(probability)
        observation = [random.choices([0, 1], [1 - p, p])[0] for p in probabilities]
        return tuple(observation)


class Model(object):
    def __init__(self, grid, R):
        self.grid = grid
        self.R = R

    def probabilityNextState(self, currState, nextState):
        if abs(nextState[0] - currState[0]) + abs(nextState[1] - currState[1]) != 1 or self.grid.isWall(nextState[0],
                                                                                                        nextState[1]):
            return 0
        neighbours = 0
        for direction in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
            x = currState[0] + direction[0]
            y = currState[1] + direction[1]
            if not self.grid.isWall(x, y):
                neighbours += 1
        assert (neighbours != 0)
        return 1 / neighbours

    def probabilityObservation(self, currState, observation):
        probability = 1
        for index, direction in enumerate([(-1, 0), (0, 1), (1, 0), (0, -1)]):
            x = currState[0] + direction[0]
            y = currState[1] + direction[1]
            while (not self.grid.isWall(x, y)):
                x += direction[0]
                y += direction[1]
            distance = abs(x - currState[0]) + abs(y - currState[1])
            if (distance >= self.R):
                alpha = 0
            else:
                alpha = 1 - ((distance - 1) / (self.R - 1))
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
                        nextBelief[nextX][nextY] += self.probabilityNextState((currX, currY), (nextX, nextY)) * \
                                                    currBelief[currX][currY]
        return nextBelief

    def updateBeliefWithObservation(self, currBelief, observation):
        nextBelief = [[0 for _ in range(self.grid.getWidth())] for _ in range(self.grid.getHeight())]
        for x in range(self.grid.getHeight()):
            for y in range(self.grid.getWidth()):
                if self.grid.isWall(x, y):
                    continue
                nextBelief[x][y] = self.probabilityObservation((x, y), observation) * currBelief[x][y]
        total = sum(sum(row) for row in nextBelief)
        assert (total != 0)
        for x in range(self.grid.getHeight()):
            for y in range(self.grid.getWidth()):
                nextBelief[x][y] /= total
        return nextBelief

    def getMostLikelyPath(self, observations):
        T = len(observations)
        M = [[[0 for _ in range(self.grid.getWidth())] for _ in range(self.grid.getHeight())] for _ in range(T + 1)]
        prev = [[[(-1, -1) for _ in range(self.grid.getWidth())] for _ in range(self.grid.getHeight())] for _ in
                range(T + 1)]
        for x in range(self.grid.getHeight()):
            for y in range(self.grid.getWidth()):
                if self.grid.isWall(x, y):
                    continue
                M[0][x][y] = 1
        total = sum(sum(row) for row in M[0])
        assert (total != 0)
        for x in range(self.grid.getHeight()):
            for y in range(self.grid.getWidth()):
                M[0][x][y] /= total
        for t in range(1, T + 1):
            for nextX in range(self.grid.getHeight()):
                for nextY in range(self.grid.getWidth()):
                    if self.grid.isWall(nextX, nextY):
                        continue
                    M[t][nextX][nextY] = 0
                    for x in range(self.grid.getHeight()):
                        for y in range(self.grid.getWidth()):
                            if self.grid.isWall(x, y):
                                continue
                            if self.probabilityNextState((x, y), (nextX, nextY)) * M[t - 1][x][y] >= M[t][nextX][nextY]:
                                M[t][nextX][nextY] = self.probabilityNextState((x, y), (nextX, nextY)) * M[t - 1][x][y]
                                prev[t][nextX][nextY] = (x, y)
                    M[t][nextX][nextY] *= self.probabilityObservation((nextX, nextY), observations[t - 1])
            total = sum(sum(row) for row in M[t])
            assert (total != 0)
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





######### model for Q1,2#########
class Gaussian(object):
    def __init__(self,mu=None,sigma=None):
        self.mu = mu
        self.sigma = sigma
        return

def incorporate_action(prev_estimate,U_t,A_t,B_t,R):
    mu_dash = (A_t @ prev_estimate.mu) + (B_t @ U_t)
    sigma_dash = (A_t @ (prev_estimate.sigma @ np.transpose(A_t))) + R
    return Gaussian(mu_dash,sigma_dash)

def incorporate_measurement(prev_estimate,U_t,Z_t,A_t,B_t,C_t,Q):
    sigma_dash = prev_estimate.sigma
    mu_dash = prev_estimate.mu
    temp = sigma_dash @ np.transpose(C_t)
    kalman_gain = temp @ np.linalg.inv(C_t @ temp + Q)
    new_estimate = Gaussian()
    new_estimate.mu = mu_dash + kalman_gain @ (Z_t - C_t @ mu_dash)
    new_estimate.sigma = (np.eye(sigma_dash.shape[0]) - kalman_gain @ C_t) @ sigma_dash
    return new_estimate


def incorporate_nonlinear_measurement(prev_estimate,Z_t,H_t,h_t,S_t):
    sigma_dash,mu_dash = prev_estimate.sigma,prev_estimate.mu
    kalman_gain = sigma_dash @ H_t.T @(1/(H_t@sigma_dash@H_t.T+S_t))
    new_estimate = Gaussian()
    new_estimate.mu= mu_dash + kalman_gain @ (Z_t - h_t)
    new_estimate.sigma = sigma_dash - kalman_gain @ H_t @ sigma_dash
    return new_estimate

class KalmanFilter(object):
    def __init__(self,Q,R):
        self.Q = Q
        self.R = R
        return
    def __call__(self,prev_estimate,U_t,Z_t,A_t,B_t,C_t,observation=True):
        '''
        :param prev_estimate: Gaussian(mean_(t), sigma_(t))
        :return new_estimate : Gaussian(mean_(t+1),sigma_(t+1))
        '''
        mu_dash = (A_t @ prev_estimate.mu)+(B_t @ U_t)
        sigma_dash = (A_t @ (prev_estimate.sigma @ np.transpose(A_t))) + self.R
        if not observation:
            new_estimate = Gaussian(mu_dash,sigma_dash)
        else:
        # print(mu_dash.shape,sigma_dash.shape,A_t.shape,B_t.shape,C_t.shape,U_t.shape,Z_t.shape)
            temp = sigma_dash @ np.transpose(C_t)
            kalman_gain = temp @ np.linalg.inv(C_t @ temp + self.Q)
            new_estimate = Gaussian()
            new_estimate.mu = mu_dash + kalman_gain @ (Z_t - C_t @ mu_dash)
            new_estimate.sigma = (np.eye(sigma_dash.shape[0]) - kalman_gain @ C_t) @ sigma_dash
        return new_estimate

class AirplaneModel(object):

    def __init__(self,X_0,Q,R,del_t=1,filter=KalmanFilter):
        '''
        state = (x,y,x_dot,y_dot)
        dim:X_0(4*1),R(4*4),Q(2*2)
        '''
        self.initial_state = X_0
        self.Q = Q
        self.R = R
        self.del_t = del_t
        self.state = self.initial_state
        self.estimated_state = Gaussian(X_0,np.eye(R.shape[0])*0.0001) #to be checked
        self.filter = filter(Q,R)
        self.landmarks = [(-100,-100),(-100,100),(100,100),(100,-100),(0,0)]#,(50,50),(50,-50),(-50,50),(-50,-50)]
        # self.landmarks = []
        self.landmark_range = 30

    def get_sensor_readings(self,X_t):
        '''
        for our observation model, Z_t = C_t*X_t + error (N(0,Q))
        C_t*X_t implies first two rows of X_t
        '''
        return np.expand_dims(np.random.multivariate_normal(mean=X_t[0:2,:].squeeze(),cov=self.Q),axis=1)

    def apply_action(self,U_t,X_t,A_t,B_t):
        '''
        X_t+1 = A_t*X_t + B_t*U_t + error(N(0,R))
        '''
        return np.expand_dims(np.random.multivariate_normal(mean=(A_t @ X_t+ B_t @ U_t).squeeze(),cov=self.R),axis=1)

    def get_landmark_info(self):
        closest_landmark = None
        dist = 1000000000
        for landmark in self.landmarks:
            d = math.sqrt((self.state[0,0]-landmark[0])*(self.state[0,0]-landmark[0])+
                          (self.state[1,0]-landmark[1])*(self.state[1,0]-landmark[1]))
            if d < dist:
                dist = d
                closest_landmark = landmark

        if dist < 30:
            return dist,closest_landmark
        else:
            return None,None

def greedy_data_assoc(estimates,observations):
    n = len(estimates)
    best_perm = None
    best_log_likelihood = -1e9
    for perm in list(permutations(range(n))):
        log_likelihood = 0
        for i in range(n):
            log_likelihood -= \
                np.transpose(observations[perm[i]] - estimates[i].mu[:2,:]) @ np.linalg.inv(estimates[i].sigma[:2,:2]) @ (observations[perm[i]] - estimates[i].mu[:2,:])

        if log_likelihood > best_log_likelihood:
            best_perm = perm
            best_log_likelihood = log_likelihood

    return best_perm

def DataAssociativeKalmanFilter(prev_estimates,U_t,Z_t,A_t,B_t,C_t,Q_t,R_t):
    '''
    :param U_t: list of U_t for each plane, similarly for each argument and return value
    '''
    new_estimates = []
    for prev_estimate,u_t,a_t,b_t,r_t in zip(prev_estimates,U_t,A_t,B_t,R_t):
        new_estimates.append(incorporate_action(prev_estimate,u_t,a_t,b_t,r_t))

    perm = greedy_data_assoc(new_estimates,Z_t)
    debug = []
    debug2 = []
    for idx in perm:
        debug.append(Z_t[idx])
    for i in range(len(perm)):
        debug2.append(prev_estimates[i].mu)
    # print(Z_t)
    # # print(prev_estimate)
    # print(debug)
    # print(debug2)
    prev_estimates = new_estimates
    new_estimates = []

    for prev_estimate,idx,u_t,a_t,b_t,c_t,q_t in zip(prev_estimates,perm,U_t,A_t,B_t,C_t,Q_t):
        new_estimates.append(incorporate_measurement(prev_estimate,u_t,Z_t[idx],a_t,b_t,c_t,q_t))

    return new_estimates



def draw_ellipse(x,y,a,b,color='orange'):
    t = np.linspace(0, 2 * np.pi, 100)
    plt.plot(x + a * np.cos(t), y + b * np.sin(t),color = color)
    plt.grid(color='lightgray', linestyle='--')


#### experiments for Q1,2 #####

from model import *
import random

def q1a():
    R = np.eye(4)
    R[2, 2] = R[3, 3] = 0.0001
    # R = np.zeros((4,4))
    Q = np.eye(2) * 100
    airplane_model = AirplaneModel(X_0=np.array([0.0, 0.0, 1.0, 1.0]).reshape((4, 1)), Q=Q, R=R)
    T = 800
    actual_states = [airplane_model.initial_state]  ## list of state objects
    estimated_states = [airplane_model.estimated_state]  ##list of Gaussians
    sensor_readings = []
    A_t = np.eye(4)
    A_t[0, 2] = A_t[1, 3] = airplane_model.del_t
    B_t = np.concatenate((np.zeros((2, 2)), np.eye(2)), axis=0)
    C_t = np.eye(2, 4)
    U_t = np.zeros((2, 1))
    for i in range(1, T + 1):
        # U_t = (np.array([np.sin(np.pi * i / 10), np.cos(np.pi * i / 10)])).reshape((2, 1))
        # print(U_t)
        airplane_model.state = airplane_model.apply_action(U_t, airplane_model.state, A_t, B_t)
        # print(airplane_model.state)
        Z_t = airplane_model.get_sensor_readings(airplane_model.state)
        sensor_readings.append(Z_t)
        estimated_state = airplane_model.filter(estimated_states[-1], U_t, Z_t, A_t, B_t, C_t)
        # print(estimated_state.sigma)
        actual_states.append(airplane_model.state)
        estimated_states.append(estimated_state)

    x = []
    y = []
    for state in actual_states:
        x.append(state[0,0])
        y.append(state[1,0])

    plt.plot(x,y,label='actual trajectory')

    x = []
    y = []
    for state in sensor_readings:
        x.append(state[0, 0])
        y.append(state[1, 0])

    plt.plot(x, y, linestyle='None', marker='x',label='observed trajectory')

    x = []
    y = []
    for state in estimated_states:
        x.append(state.mu[0, 0])
        y.append(state.mu[1, 0])

    plt.plot(x, y, label='Estimated trajectory')


    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()


def q1bc():
    R = np.eye(4)
    R[2, 2] = R[3, 3] = 0.0001
    # R = np.zeros((4,4))
    Q = np.eye(2) * 100
    airplane_model = AirplaneModel(X_0=np.array([0.0, 0.0, 0.0, 0.0]).reshape((4, 1)), Q=Q, R=R)
    T = 200
    actual_states = [airplane_model.initial_state]  ## list of state objects
    estimated_states = [airplane_model.estimated_state]  ##list of Gaussians
    sensor_readings = []
    A_t = np.eye(4)
    A_t[0, 2] = A_t[1, 3] = airplane_model.del_t
    B_t = np.concatenate((np.zeros((2, 2)), np.eye(2)), axis=0)
    C_t = np.eye(2, 4)
    U_t = np.zeros((2, 1))
    for i in range(1, T + 1):
        U_t = (np.array([np.sin(np.pi * i / 10), np.cos(np.pi * i / 10)])).reshape((2, 1))
        # print(U_t)
        airplane_model.state = airplane_model.apply_action(U_t, airplane_model.state, A_t, B_t)
        # print(airplane_model.state)
        Z_t = airplane_model.get_sensor_readings(airplane_model.state)
        sensor_readings.append(Z_t)
        estimated_state = airplane_model.filter(estimated_states[-1], U_t, Z_t, A_t, B_t, C_t)
        # print(estimated_state.sigma)
        actual_states.append(airplane_model.state)
        estimated_states.append(estimated_state)

    x = []
    y = []
    for state in actual_states:
        x.append(state[0,0])
        y.append(state[1,0])

    plt.plot(x,y,label='Actual trajectory')

    x = []
    y = []
    cnt = 0
    for state in estimated_states:
        if cnt%10 == 0:
            draw_ellipse(state.mu[0, 0], state.mu[1, 0], np.sqrt(state.sigma[0, 0]), np.sqrt(state.sigma[1, 1]))
        x.append(state.mu[0, 0])
        y.append(state.mu[1, 0])
        cnt = cnt + 1

    plt.plot(x, y,label='Estimated trajectory')

    x = []
    y = []
    for state in sensor_readings:
        x.append(state[0, 0])
        y.append(state[1, 0])

    plt.plot(x, y, label='Observed trajectory', marker='x',linestyle='None')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()


def q1d():
    R = np.eye(4)
    R[2, 2] = R[3, 3] = 0.0001
    # R = np.zeros((4,4))
    Q = np.eye(2) * 100
    airplane_model = AirplaneModel(X_0=np.array([0.0, 0.0, 1.0, 1.0]).reshape((4, 1)), Q=Q, R=R)
    T = 200
    actual_states = [airplane_model.initial_state]  ## list of state objects
    estimated_states = [airplane_model.estimated_state]  ##list of Gaussians
    sensor_readings = []
    A_t = np.eye(4)
    A_t[0, 2] = A_t[1, 3] = airplane_model.del_t
    B_t = np.concatenate((np.zeros((2, 2)), np.eye(2)), axis=0)
    C_t = np.eye(2, 4)
    for i in range(1, T + 1):
        U_t = (np.array([np.sin(np.pi * i / 10), np.cos(np.pi * i / 10)])).reshape((2, 1))
        # print(U_t)
        airplane_model.state = airplane_model.apply_action(U_t, airplane_model.state, A_t, B_t)
        # print(airplane_model.state)
        Z_t = airplane_model.get_sensor_readings(airplane_model.state)
        sensor_readings.append(Z_t)
        observation = not ((i >= 10 and i < 30) or (i >= 60 and i < 80))
        estimated_state = airplane_model.filter(estimated_states[-1], U_t, Z_t, A_t, B_t, C_t, observation=observation)
        print(estimated_state.sigma)
        actual_states.append(airplane_model.state)
        estimated_states.append(estimated_state)

    x = []
    y = []
    for state in actual_states:
        x.append(state[0,0])
        y.append(state[1,0])

    plt.plot(x,y,label='Actual trajectory',marker='.',color='blue')

    x = []
    y = []
    cnt = 0
    for state in estimated_states:
        if cnt % 5 == 0:
            if ((cnt > 10 and cnt <= 30) or (cnt > 60 and cnt <= 80)):
                plt.plot(state.mu[0, 0], state.mu[1, 0], marker='x',color='black')
                draw_ellipse(state.mu[0, 0], state.mu[1, 0], np.sqrt(state.sigma[0, 0]), np.sqrt(state.sigma[1, 1]),color='red')
            else:
                draw_ellipse(state.mu[0, 0], state.mu[1, 0], np.sqrt(state.sigma[0, 0]), np.sqrt(state.sigma[1, 1]))
        x.append(state.mu[0, 0])
        y.append(state.mu[1, 0])
        cnt = cnt + 1

    plt.plot(x, y,label='Estimated trajectory',color='green')

    # x = []
    # y = []
    # for state in sensor_readings:
    #     x.append(state[0, 0])
    #     y.append(state[1, 0])
    #
    # plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()


def q1e():

    R = np.eye(4)
    R[2, 2] = R[3, 3] = 0.01
    # R = np.zeros((4,4))
    Q = np.eye(2) * 100
    airplane_model = AirplaneModel(X_0=np.array([0.0, 0.0, 0.0, 0.0]).reshape((4, 1)), Q=Q, R=R)
    T = 200
    actual_states = [airplane_model.initial_state]  ## list of state objects
    estimated_states = [airplane_model.estimated_state]  ##list of Gaussians
    sensor_readings = []
    A_t = np.eye(4)
    A_t[0, 2] = A_t[1, 3] = airplane_model.del_t
    B_t = np.concatenate((np.zeros((2, 2)), np.eye(2)), axis=0)
    C_t = np.eye(2, 4)
    U_t = np.zeros((2, 1))
    for i in range(1, T + 1):
        U_t = (np.array([np.sin(np.pi * i / 10), np.cos(np.pi * i / 10)])).reshape((2, 1))
        # print(U_t)
        airplane_model.state = airplane_model.apply_action(U_t, airplane_model.state, A_t, B_t)
        print(airplane_model.state[2:, 0])
        Z_t = airplane_model.get_sensor_readings(airplane_model.state)
        sensor_readings.append(Z_t)
        observation = not ((i >= 10 and i < 30) or (i >= 60 and i < 80))
        estimated_state = airplane_model.filter(estimated_states[-1], U_t, Z_t, A_t, B_t, C_t, observation=observation)
        # print(estimated_state.sigma)
        actual_states.append(airplane_model.state)
        estimated_states.append(estimated_state)

    x = []
    y = []
    for state in actual_states:
        x.append(state[2, 0])
        y.append(state[3, 0])

    ax = plt.axes(projection='3d')
    # Data for a three-dimensional line
    z = [i for i in range(201)]
    ax.plot3D(x, y, z)

    x = []
    y = []
    for state in estimated_states:
        x.append(state.mu[2, 0])
        y.append(state.mu[3, 0])

    ax.plot3D(x, y, z)

    # x = []
    # y = []
    # for state in sensor_readings:
    #     x.append(state[0, 0])
    #     y.append(state[1, 0])
    #
    # plt.plot(x, y)
    plt.show()


def q1f():

    R = np.eye(4)
    R[2, 2] = R[3, 3] = 0.01
    # R = np.zeros((4,4))
    Q = np.eye(2) * 100
    num_planes = 2
    airplane_models = [AirplaneModel
                       (X_0=np.concatenate(
        (np.random.uniform(0,25,size=(2,1)),np.random.uniform(0,1,size=(2,1))),axis=0), Q=Q, R=R) for i in range(num_planes)]

    T = 200
    actual_states = [[airplane_models[i].initial_state for i in range(num_planes)]]  ## list of state objects
    associated_estimated_states = [[airplane_models[i].estimated_state for i in range(num_planes)]]  ##list of Gaussians
    sensor_readings = []
    A_t = np.eye(4)
    A_t[0, 2] = A_t[1, 3] = 1
    B_t = np.concatenate((np.zeros((2, 2)), np.eye(2)), axis=0)
    C_t = np.eye(2, 4)
    U_t = np.zeros((2, 1))

    A_t, B_t, C_t = [A_t for i in range(num_planes)], [B_t for i in range(num_planes)], [C_t for i in range(num_planes)]
    Q, R = [Q for i in range(num_planes)], [R for i in range(num_planes)]

    for i in range(1, T + 1):
        U_t = [(np.array([np.sin(np.pi * i / 10), np.cos(np.pi * i / 10)])).reshape((2, 1)) for j in range(num_planes)]
        # print(U_t)
        Z_t = []
        next_actual_states = []
        for i in range(num_planes):
            airplane_model = airplane_models[i]
            airplane_model.state = airplane_model.apply_action(U_t[i], airplane_model.state, A_t[i], B_t[i])
            #print(airplane_model.state[2:, 0])
            next_actual_states.append(airplane_model.state)
            z_t = airplane_model.get_sensor_readings(airplane_model.state)
            Z_t.append(z_t)

        random.shuffle(Z_t)
        # print(Z_t)
        new_estimations = DataAssociativeKalmanFilter(associated_estimated_states[-1], U_t, Z_t, A_t, B_t, C_t, Q, R)
        associated_estimated_states.append(new_estimations)
        actual_states.append(next_actual_states)

    for i in range(num_planes):
        x = []
        y = []
        for states in actual_states:
            x.append(states[i][0, 0])
            y.append(states[i][1, 0])
        plt.plot(x,y,label='actual trajectory')

        x = []
        y = []

        for states in associated_estimated_states:
            x.append(states[i].mu[0, 0])
            y.append(states[i].mu[1, 0])

        plt.plot(x, y, label='estimated trajectory')
    # x = []
    # y = []
    # for state in sensor_readings:
    #     x.append(state[0, 0])
    #     y.append(state[1, 0])
    #
    # plt.plot(x, y)
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def q2c():

    R = np.eye(4)*0.0001
    # R = np.zeros((4,4))
    Q = np.eye(2) * 100
    airplane_model = AirplaneModel(X_0=np.array([30,-70,0.1,0.1]).reshape((4, 1)), Q=Q, R=R)
    for landmark in airplane_model.landmarks:
        plt.plot(landmark[0],landmark[1],marker='x',color='red')
    T = 800
    actual_states = [airplane_model.initial_state]  ## list of state objects
    estimated_states = [airplane_model.estimated_state]  ##list of Gaussians
    sensor_readings = []
    markers = [False,]
    A_t = np.eye(4)
    A_t[0, 2] = A_t[1, 3] = airplane_model.del_t
    B_t = np.concatenate((np.zeros((2, 2)), np.eye(2)), axis=0)
    C_t = np.eye(2, 4)
    S_t = np.eye(1)*1
    for i in range(1, T + 1):
        # print(U_t)
        U_t = (np.array([-1*np.sin(np.pi * i / 10), -1*np.cos(np.pi * i / 10)])).reshape((2, 1))
        U_t = (np.array([-0.001*airplane_model.state[0,0],-0.0005*airplane_model.state[1,0]])).reshape((2, 1))
        airplane_model.state = airplane_model.apply_action(U_t, airplane_model.state, A_t, B_t)
        actual_states.append(airplane_model.state)
        # print(airplane_model.state)
        Z_t = airplane_model.get_sensor_readings(airplane_model.state)
        sensor_readings.append(Z_t)
        dist,landmark = airplane_model.get_landmark_info()
        if landmark == None:
            markers.append(False)
            estimated_state = airplane_model.filter(estimated_states[-1], U_t, Z_t, A_t, B_t, C_t)
            estimated_states.append(estimated_state)
        else:
            markers.append(True)
            estimated_state = airplane_model.filter(estimated_states[-1], U_t, Z_t, A_t, B_t, C_t)
            dist = np.array([dist]).reshape((1,1))
            h_t = math.sqrt((estimated_state.mu[0,0]-landmark[0])*(estimated_state.mu[0,0]-landmark[0])
                            +(estimated_state.mu[1,0]-landmark[1])*(estimated_state.mu[1,0]-landmark[1]))
            H_t = np.array([(estimated_state.mu[0,0]-landmark[0])/h_t,(estimated_state.mu[1,0]-landmark[1])/h_t,0,0]).reshape((1,4))
            h_t = np.array(h_t).reshape((1,1))
            print(h_t,dist)
            estimated_state = incorporate_nonlinear_measurement(estimated_state,dist,H_t,h_t,S_t)
            estimated_states.append(estimated_state)

    x = []
    y = []
    for state in actual_states:
        x.append(state[0,0])
        y.append(state[1,0])

    plt.plot(x,y,label='Actual trajectory')

    x = []
    y = []
    cnt = 0
    for state in estimated_states:
        if cnt%5 == 0:
            if markers[cnt]:
                draw_ellipse(state.mu[0, 0], state.mu[1, 0], np.sqrt(state.sigma[0, 0]), np.sqrt(state.sigma[1, 1]),color='red')
            else:
                draw_ellipse(state.mu[0, 0], state.mu[1, 0], np.sqrt(state.sigma[0, 0]), np.sqrt(state.sigma[1, 1]))
        x.append(state.mu[0, 0])
        y.append(state.mu[1, 0])
        cnt = cnt + 1

    plt.plot(x, y,label='Estimated trajectory')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    # x = []
    # y = []
    # for state in sensor_readings:
    #     x.append(state[0, 0])
    #     y.append(state[1, 0])
    #
    # plt.plot(x, y)
    plt.show()


### experiments for Q3 #####
import matplotlib as mpl
import matplotlib.pyplot as plt
import statistics

R = 5
LAYOUT = 'layouts/sample.lay'


def q3a():
    grid = Grid(LAYOUT)
    robot = Robot(grid, R)
    model = Model(grid, R)

    cmap = mpl.colors.ListedColormap(['black', 'white', 'red', 'yellow', 'green'])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

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

        gridData[robot.getPosition()[0]][robot.getPosition()[1]] += 1
        gridData[estimatedPosition[0]][estimatedPosition[1]] += 2
        plt.imshow(gridData, cmap=cmap, norm=norm)
        plt.pause(0.1)
        gridData[robot.getPosition()[0]][robot.getPosition()[1]] -= 1
        gridData[estimatedPosition[0]][estimatedPosition[1]] -= 2

    plt.show()

    mostLikelyPath = model.getMostLikelyPath(observations)
    print(path)
    print(mostLikelyPath)

    figure, axis = plt.subplots(2)

    X = [p[0] for p in path]
    Y = [p[1] for p in path]
    axis[0].plot(Y, X)
    # axis[0].grid()
    axis[0].imshow(gridData, cmap=cmap, norm=norm)

    X = [p[0] for p in mostLikelyPath]
    Y = [p[1] for p in mostLikelyPath]
    axis[1].plot(Y, X, 'y')
    # axis[1].grid()
    axis[1].imshow(gridData, cmap=cmap, norm=norm)

    plt.show()


def q3b():
    grid = Grid(LAYOUT)
    robot = Robot(grid, R)
    model = Model(grid, R)

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

    for t in range(1, 26):
        if t != 1:
            robot.move()
            belief = model.updateBeliefWithTime(belief)

        for x in range(grid.getHeight()):
            for y in range(grid.getWidth()):
                if not grid.isWall(x, y):
                    gridData[x][y] = 1 - belief[x][y]

        # plt.imshow(gridData, cmap='gray')
        # plt.pause(0.5)

        observation = robot.getObservation()
        belief = model.updateBeliefWithObservation(belief, observation)

        for x in range(grid.getHeight()):
            for y in range(grid.getWidth()):
                if not grid.isWall(x, y):
                    gridData[x][y] = 1 - belief[x][y]

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

            positionError += abs(robot.getPosition()[0] - estimatedPosition[0]) + abs(
                robot.getPosition()[1] - estimatedPosition[1])
        mostLikelyPath = model.getMostLikelyPath(observations)
        for i in range(len(mostLikelyPath)):
            mostLikelyPathError += abs(path[i][0] - mostLikelyPath[i][0]) + abs(path[i][1] - mostLikelyPath[i][1])
        estimatedPathErrors.append(positionError)
        mostLikelyPathErrors.append(mostLikelyPathError)

    print('Position Estimation Error Mean:', statistics.mean(estimatedPathErrors))
    print('Position Estimation Error Std Dev:', statistics.stdev(estimatedPathErrors))
    print('Most Likely Path Error Mean:', statistics.mean(mostLikelyPathErrors))
    print('Most Likely Path Error Std Dev:', statistics.stdev(mostLikelyPathErrors))

q3a()