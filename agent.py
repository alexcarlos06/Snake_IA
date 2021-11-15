import torch
import random
import numpy as np
from collections import deque
from Snake_IA.snake_game import SnakeGameAI, Direction, Point, BLOCK_SIZE

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gama = 0  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = None  # TODO: model
        self.trainer = None  # TODO: trainer

    def get_state(self, game):
        # Pegando as coordenadas de acordo com a cabeça da cobra
        head = game.snake[0]
        point_l = Point(head - BLOCK_SIZE, head.y)
        point_r = Point(head + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        # Obtendo a direção atual em que a cobra está se movendo
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # Criando as 11 variáveis de estado possivel
        state = [
            # PERIGO
            # Perigo mantendo a mesma direção (Danger straight)
            (dir_r and game.is_collision(point_r)) or (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or (dir_d and game.is_collision(point_d)),

            # Perigo virando no sentido horário (Danger right)
            (dir_u and game.is_collision(point_r)) or (dir_r and game.is_collision(point_d)) or
            (dir_d and game.is_collision(point_l)) or (dir_l and game.is_collision(point_u)),

            # Perigo virando no sentido antihorario (Danger left)
            (dir_u and game.is_collision(point_l)) or (dir_l and game.is_collision(point_d)) or
            (dir_d and game.is_collision(point_r)) or (dir_r and game.is_collision(point_u)),

            # MOVIMENTO (Move)
            dir_u, dir_r, dir_d, dir_l,

            # LOCALIZAÇÃO DA COMIDA (Food Location)
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y,  # food down
        ]
        return np.array(state, dtype=int)

    def remeber(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model.predict(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move


def train():
    plot_score = []
    plot_mean_score = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()

    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remeber(state_old, final_move, reward, state_new, done)

        if done:
            # (Memoria de repetição ou memoria de experiência)
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                # agent.model.save()

            print('Game ', agent.n_games, 'Score: ', score, 'Record: ', record)
            # TODO: plot


if __name__ == '__main__':
    train()
