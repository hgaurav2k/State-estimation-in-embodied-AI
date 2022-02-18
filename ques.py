from model import *
import random

def q1bc():
    R = np.eye(4)
    R[2, 2] = R[3, 3] = 0.01
    # R = np.zeros((4,4))
    Q = np.eye(2) * 100
    airplane_model = AirplaneModel(X_0=np.array([0.0, 0.0, 0.0, 0.0]).reshape((4, 1)), Q=Q, R=R)
    T = 20
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
        print(estimated_state.sigma)
        actual_states.append(airplane_model.state)
        estimated_states.append(estimated_state)

    # x = []
    # y = []
    # for state in actual_states:
    #     x.append(state[0,0])
    #     y.append(state[1,0])
    #
    # plt.plot(x,y)

    x = []
    y = []
    cnt = 0
    for state in estimated_states:
        if cnt%10 == 0:
            draw_ellipse(state.mu[0, 0], state.mu[1, 0], np.sqrt(state.sigma[0, 0]), np.sqrt(state.sigma[1, 1]))
        x.append(state.mu[0, 0])
        y.append(state.mu[1, 0])
        cnt = cnt + 1

    plt.plot(x, y)

    # x = []
    # y = []
    # for state in sensor_readings:
    #     x.append(state[0, 0])
    #     y.append(state[1, 0])
    #
    # plt.plot(x, y)
    plt.show()


def q1d():
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
        # print(airplane_model.state)
        Z_t = airplane_model.get_sensor_readings(airplane_model.state)
        sensor_readings.append(Z_t)
        observation = not ((i >= 10 and i < 30) or (i >= 60 and i < 80))
        estimated_state = airplane_model.filter(estimated_states[-1], U_t, Z_t, A_t, B_t, C_t, observation=observation)
        print(estimated_state.sigma)
        actual_states.append(airplane_model.state)
        estimated_states.append(estimated_state)

    # x = []
    # y = []
    # for state in actual_states:
    #     x.append(state[0,0])
    #     y.append(state[1,0])
    #
    # plt.plot(x,y)

    x = []
    y = []
    cnt = 0
    for state in estimated_states:
        if cnt % 5 == 0:
            if ((cnt > 10 and cnt <= 30) or (cnt > 60 and cnt <= 80)):
                plt.plot(state.mu[0, 0], state.mu[1, 0], marker='x')
            draw_ellipse(state.mu[0, 0], state.mu[1, 0], np.sqrt(state.sigma[0, 0]), np.sqrt(state.sigma[1, 1]))
        x.append(state.mu[0, 0])
        y.append(state.mu[1, 0])
        cnt = cnt + 1

    plt.plot(x, y)

    # x = []
    # y = []
    # for state in sensor_readings:
    #     x.append(state[0, 0])
    #     y.append(state[1, 0])
    #
    # plt.plot(x, y)
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
    num_planes = 3
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

        new_estimations = DataAssociativeKalmanFilter(associated_estimated_states[-1], U_t, Z_t, A_t, B_t, C_t, Q, R)
        associated_estimated_states.append(new_estimations)
        actual_states.append(next_actual_states)

    for i in range(num_planes):
        x = []
        y = []
        for states in actual_states:
            x.append(states[i][0, 0])
            y.append(states[i][1, 0])
        plt.plot(x,y)

        x = []
        y = []

        for states in associated_estimated_states:
            x.append(states[i].mu[0, 0])
            y.append(states[i].mu[1, 0])

        plt.plot(x,y)
    # x = []
    # y = []
    # for state in sensor_readings:
    #     x.append(state[0, 0])
    #     y.append(state[1, 0])
    #
    # plt.plot(x, y)
    plt.show()


if __name__ == '__main__':
    q1f()