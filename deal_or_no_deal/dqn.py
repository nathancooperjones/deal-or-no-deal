# -*- coding: utf-8 -*-
from collections import deque
import random

import gym
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
import numpy as np

import deal_or_no_deal  # noqa: F401


EPISODES = 25000


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.999
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()

        model.add(Dense(self.state_size,
                        input_dim=self.state_size,
                        kernel_initializer='glorot_normal',
                        activation='relu'))
        model.add(Dense(42, kernel_initializer='glorot_normal', activation='relu'))
        model.add(Dense(16, kernel_initializer='glorot_normal', activation='relu'))
        model.add(Dense(8, kernel_initializer='glorot_normal', activation='relu'))
        model.add(Dense(self.action_size, kernel_initializer='glorot_normal', activation='linear'))

        model.compile(loss='mse', optimizer=Adam(learning_rate=0.0001))

        return model

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        act_values = self.model.predict(state)

        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == '__main__':
    env = gym.make('deal-or-no-deal-v0')
    # state_size = env.observation_space[].shape[0]
    state_size = 27
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    # agent.load("./save/cartpole-dqn.h5")
    done = False
    batch_size = 32

    for e in range(EPISODES):
        # state_list = list()
        # action_list = list()
        # next_state_list = list()
        # done_list = list()

        state = env.reset()
        # TODO: FIX
        state = np.concatenate([state[0][0], state[0][2] / 9])
        state = np.reshape(state, [1, state_size])
        for time in range(500):
            # env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            # TODO: FIX
            next_state = np.concatenate([next_state[0], next_state[2] / 9])
            next_state = np.reshape(next_state, [1, state_size])
            agent.memorize(state, action, reward, next_state, done)
            # state_list.append(state)
            # action_list.append(action)
            # next_state_list.append(next_state)
            # done_list.append(done)

            state = next_state
            if done:
                # for idx in range(len(state_list)):
                #     agent.memorize(state_list[idx],
                #                    action_list[idx],
                #                    reward,
                #                    next_state_list[idx],
                #                    done_list[idx])
                print('Episode: {}/{}, length: {}, score: {}, e: {:.2}'
                      .format(e, EPISODES, time, reward, agent.epsilon))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        if (e + 1) % 100 == 0:
            agent.save('/deal_or_no_deal/data/dqn.h5')
