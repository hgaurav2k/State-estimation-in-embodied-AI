import numpy as np
import matplotlib.pyplot as plt
from itertools import  permutations

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

    sigma_dash = prev_estimate.sigma
    mu_dash = prev_estimate.mu
    temp = sigma_dash @ np.transpose(H_t)
    kalman_gain = temp @ np.linalg.inv(H_t @ temp + S_t)
    new_estimate = Gaussian()
    new_estimate.mu = mu_dash + kalman_gain @ (Z_t - h_t)
    new_estimate.sigma = (np.eye(sigma_dash.shape[0]) - kalman_gain @ H_t) @ sigma_dash
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


class ExtendedKalmanFilter(object):

    def __init__(self):
        return
    def __call__(self,prev_estimate,U_t,Z_t,A_t,B_t,H_t,h_t,Q_t,R_t):
        '''
        h_t and H_t(h_t^') are functions since they need to be evaluated
        on mu_dash which is not available yet
        '''
        new_estimate = incorporate_action(prev_estimate,U_t,A_t,B_t,R_t)





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
        self.landmarks = [(-100,-100),(-100,100),(100,100),(100,-100),(0,0)]
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
        def dist(x):
            return np.sqrt((self.state[0,0]-x[0])**2+(self.state[1,0]-x[1])**2)
        min_dist = 1e9
        closest  = -1
        for landmark in self.landmarks:
            if dist(landmark) < min_dist:
                closest = landmark
                min_dist = dist(landmark)

        if min_dist <= 30:
            return min_dist+np.random.normal(0,1),closest
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
    :param U_t: list of U_t for each plane, similarily for each argument and return value
    '''
    new_estimates = []
    for prev_estimate,u_t,a_t,b_t,r_t in zip(prev_estimates,U_t,A_t,B_t,R_t):
        new_estimates.append(incorporate_action(prev_estimate,u_t,a_t,b_t,r_t))

    perm = greedy_data_assoc(new_estimates,Z_t)

    prev_estimates = new_estimates
    new_estimates = []

    for prev_estimate,idx,u_t,a_t,b_t,c_t,q_t in zip(prev_estimates,perm,U_t,A_t,B_t,C_t,Q_t):
        new_estimates.append(incorporate_measurement(prev_estimate,u_t,Z_t[idx],a_t,b_t,c_t,q_t))

    return new_estimates



def draw_ellipse(x,y,a,b):
    t = np.linspace(0, 2 * np.pi, 100)
    plt.plot(x + a * np.cos(t), y + b * np.sin(t),color = 'orange')
    plt.grid(color='lightgray', linestyle='--')


