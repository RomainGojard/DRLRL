import numpy as np

class HoverEnv:
    """
    Environnement simplifié pour simuler une voiture Rocket League
    qui essaie de se maintenir en l'air entre deux hauteurs cibles.
    """
    def __init__(self,
                 g=0.004,           # gravité vers le bas
                 thrust=0.01,       # poussée vers le haut
                 pos_min=0.0,       # limite bas de l'arène
                 pos_max=1.0,       # limite haut de l'arène
                 vel_min=-0.3,      # vitesse minimale
                 vel_max=0.3,       # vitesse maximale
                 max_steps=400,     # nombre max de pas par épisode
                 target_center=0.5, # centre de la zone cible
                 target_halfwidth=0.1): # demi-largeur de la zone cible
        self.g = g
        self.thrust = thrust
        self.pos_min = pos_min
        self.pos_max = pos_max
        self.vel_min = vel_min
        self.vel_max = vel_max
        self.max_steps = max_steps
        self.target_center = target_center
        self.target_halfwidth = target_halfwidth
        self.reset()

    def reset(self):
        """Réinitialise l'environnement au début d'un épisode."""
        self.pos = 0.2 + np.random.uniform(-0.02, 0.02)  # position initiale
        self.vel = 0.0                                   # vitesse initiale
        self.steps = 0
        return (self.pos, self.vel)

    def step(self, action):
        """
        Fait évoluer l'environnement d'un pas de temps.
        action = 0 : pas de poussée
        action = 1 : poussée vers le haut
        """
        # force appliquée
        if action == 1:
            accel = self.thrust - self.g
        else:
            accel = -self.g  # seulement la gravité

        # mise à jour de la vitesse et position
        self.vel += accel
        self.vel = np.clip(self.vel, self.vel_min, self.vel_max)
        self.pos += self.vel
        self.steps += 1

        # vérifier si l'épisode est terminé
        done = False
        reward = 0
        if self.pos <= self.pos_min or self.pos >= self.pos_max:
            # sortie des limites = crash
            done = True
            reward = -10
        else:
            # donner une récompense si la voiture est dans la zone cible
            if abs(self.pos - self.target_center) <= self.target_halfwidth:
                reward = 1.0
            else:
                reward = -1.0

        if self.steps >= self.max_steps:
            done = True

        return (self.pos, self.vel), reward, done

    def state_to_indices(self, pos, vel, n_pos, n_vel):
        """Convertit un état (position, vitesse) en indices de tableau Q."""
        pos_norm = (pos - self.pos_min) / (self.pos_max - self.pos_min)
        vel_norm = (vel - self.vel_min) / (self.vel_max - self.vel_min)
        i_pos = int(np.clip(pos_norm * (n_pos - 1), 0, n_pos - 1))
        i_vel = int(np.clip(vel_norm * (n_vel - 1), 0, n_vel - 1))
        return i_pos, i_vel
