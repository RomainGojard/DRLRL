import numpy as np

class QLearningAgent3D:
    def __init__(self,
                 n_x, n_y, n_z,
                 n_vx, n_vy, n_vz,
                 n_theta, n_phi, n_psi,
                 n_omega_x, n_omega_y, n_omega_z,
                 n_actions,
                 alpha=0.1, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.999):
        # Discrétisation
        self.n_x, self.n_y, self.n_z = n_x, n_y, n_z
        self.n_vx, self.n_vy, self.n_vz = n_vx, n_vy, n_vz
        self.n_theta, self.n_phi, self.n_psi = n_theta, n_phi, n_psi
        self.n_omega_x, self.n_omega_y, self.n_omega_z = n_omega_x, n_omega_y, n_omega_z
        self.n_actions = n_actions

        # Hyperparamètres
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Table Q géante (⚠️ mémoire ! → float32)
        self.q = np.zeros((n_x, n_y, n_z,
                           n_vx, n_vy, n_vz,
                           n_theta, n_phi, n_psi,
                           n_omega_x, n_omega_y, n_omega_z,
                           n_actions),
                          dtype=np.float32)

    def choose_action(self, s_idx):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        qvals = self.q[s_idx]
        max_q = np.max(qvals)
        best_actions = np.flatnonzero(qvals == max_q)
        return int(np.random.choice(best_actions))

    def update(self, s_idx, a, r, s2_idx, done):
        q_predict = self.q[s_idx][a]
        if done:
            q_target = r
        else:
            q_target = r + self.gamma * np.max(self.q[s2_idx])
        self.q[s_idx][a] += self.alpha * (q_target - q_predict)

        # Décroissance epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
