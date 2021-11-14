import pygame
import random
import numpy as np
from enum import Enum
from collections import namedtuple

pygame.init()


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple('Point', ('x', 'y'))
font = pygame.font.Font(r'C:\Users\alexc\Documents\Portifolio\MachineLearng\Snake_IA\arial.ttf', 25)
# rgb colors

WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 10


class SnakeGameAI:

    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h

        # init dysplay
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()

        # Declaração de variáveis de inicialização
        self.direction = 0
        self.score = 0
        self.frame_iteration = 0
        self.speed = 0
        self.head = None
        self.snake = None
        self.food = None

        # Iniciando o jogo
        self.reset()

    def reset(self):
        # init game state

        self.direction = Direction.RIGHT

        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self.speed = SPEED
        self._place_food()
        self.frame_iteration = 0

    def play_step(self, action):
        # 1. colect user input
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. move
        self._move(action)
        self.snake.insert(0, self.head)
        # 3. check if game over
        reward = 0
        game_over = False
        if self._is_collision() or self.frame_iteration > len(self.snake) * 100:
            game_over = True
            reward -= 10
            return reward, game_over, self.score
        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
        # 5. update ui and clock
        self._update_ui()
        self._update_speed()
        self.clock.tick(self.speed)
        # 6. return game ove and score

        return reward, game_over, self.score

    def _is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        # Verificar se a cabeça da cobra atingiu algum dos limetes da tela
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y < 0 or pt.y > self.h - BLOCK_SIZE:
            return True
        # hits itself
        # Verificar se a cabeça da cobra atingiu alguma de suas outras partes
        if pt in self.snake[1:]:
            return True

    def _update_ui(self):
        self.display.fill(BLACK)

        for point in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(point.x, point.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(point.x + 4, point.y + 4, 12, 12))
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        text = font.render('Score: ' + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        # [straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # Neste caso nada acontece
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # Neste caso vira para a direita r-> d-> l-> u
        else:  # [0, 0, 1]
            previous_idx = (idx - 1) % 4
            new_dir = clock_wise[previous_idx]  # Neste caso vira para a esquerda r-> u-> l-> d

        self.direction = new_dir
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        self.head = Point(x, y)

    def _update_speed(self):
        if self.score > self.speed:
            self.speed += 0.5

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

