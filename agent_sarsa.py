import numpy as np

class SARSAAgent:
    def __init__(self, n_pos, n_vel, n_actions,
                 alpha=0.1, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.999):
        self.n_pos = n_pos
        self.n_vel = n_vel
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q = np.zeros((n_pos, n_vel, n_actions))

    def choose_action(self, s_idx):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.q[s_idx[0], s_idx[1], :]))

    def update(self, s_idx, a, r, s2_idx, a2, done):
        q_predict = self.q[s_idx[0], s_idx[1], a]
        if done:
            q_target = r
        else:
            q_target = r + self.gamma * self.q[s2_idx[0], s2_idx[1], a2]
        self.q[s_idx[0], s_idx[1], a] += self.alpha * (q_target - q_predict)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay