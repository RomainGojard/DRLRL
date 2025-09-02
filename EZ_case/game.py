import pygame
import sys
import numpy as np

WHITE = (255,255,255)
RED = (200,50,50)
GREEN = (50,200,50)
BLUE = (50,50,200)
BLACK = (0,0,0)

class HoverGame:
    def __init__(self, env, agent, actions):
        pygame.init()
        self.env = env
        self.agent = agent
        self.actions = actions

        self.WIDTH, self.HEIGHT = 200, 400
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Hover Q-learning")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 24)

        self.reset()

    def reset(self):
        self.state = self.env.reset()
        self.s_idx = self.env.state_to_indices(self.state[0], self.state[1],
                                               self.agent.n_pos, self.agent.n_vel)
        self.score = 0

    def render_loop(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # action gloutonne
            a_idx = np.argmax(self.agent.q[self.s_idx[0], self.s_idx[1], :])
            action = self.actions[a_idx]
            (pos, vel), r, done = self.env.step(action)
            self.s_idx = self.env.state_to_indices(pos, vel, self.agent.n_pos, self.agent.n_vel)
            self.score += r

            # dessin
            self.screen.fill(WHITE)
            y_center = int(self.env.target_center * self.HEIGHT)
            y_half = int(self.env.target_halfwidth * self.HEIGHT)
            pygame.draw.line(self.screen, GREEN, (0, y_center-y_half), (self.WIDTH, y_center-y_half), 2)
            pygame.draw.line(self.screen, GREEN, (0, y_center+y_half), (self.WIDTH, y_center+y_half), 2)

            y_pos = int((1 - pos) * self.HEIGHT)

            car_rect = pygame.Rect(self.WIDTH//2 - 10, y_pos-10, 20, 20)
            pygame.draw.rect(self.screen, BLUE, car_rect)

            # flèche de poussée
            if action == 1:
                pygame.draw.polygon(self.screen, RED, [
                (self.WIDTH//2, y_pos+20),
                (self.WIDTH//2-5, y_pos+30),
                (self.WIDTH//2+5, y_pos+30)
            ])

            score_text = self.font.render(f"Score: {int(self.score)}", True, BLACK)
            self.screen.blit(score_text, (10, 10))

            pygame.display.flip()
            self.clock.tick(30)

        pygame.quit()
        sys.exit()
