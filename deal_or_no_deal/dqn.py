from collections import deque
import random

import fire
import gym
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
import numpy as np

import deal_or_no_deal  # noqa: F401


class DQNAgent:
    """
    Deep Q Neural Network agent.

    Parameters
    ----------
    state_size: int
        Dimension of the input to the model
    action_size: int
        Dimension of the output to the model
    memory_size: int
        Maximum length of the memory queue to hold at once (default 500)
    gamma: float
        Discount rate of the reward prediction (default 0.95)
    epsilon_decay: float
        Amount to decay `epsilon` by after each episode (default 0.9995)

    """
    def __init__(self, state_size, action_size, memory_size=500, gamma=0.95, epsilon_decay=0.9999):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.001
        self.epsilon_decay = epsilon_decay
        self.model = self._build_model()

    def _build_model(self):
        """Build the Keras model."""
        model = Sequential()

        model.add(Dense(self.state_size,
                        input_dim=self.state_size,
                        kernel_initializer='glorot_normal',
                        activation='relu'))
        model.add(Dense(42, kernel_initializer='glorot_normal', activation='relu'))
        model.add(Dense(16, kernel_initializer='glorot_normal', activation='relu'))
        model.add(Dense(8, kernel_initializer='glorot_normal', activation='relu'))
        model.add(Dense(self.action_size, kernel_initializer='glorot_normal', activation='linear'))

        model.compile(loss='mse', optimizer=Adam(learning_rate=0.0005))

        return model

    def memorize(self, state, action, reward, next_state, done):
        """Memorize a given state, action, reward, next_state, and done for recall in the future."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Get a prediction from a model based on the current state, if not still exploring."""
        if np.random.rand() <= self.epsilon:
            if state[0][-1] * 10 > 2:
                return random.randrange(self.action_size)
            else:
                return 1

        act_values = self.model.predict(state)

        # return action with highest probability from model
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        """Replay a state from memory and fit the model on the reward."""
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

    def save(self, filename):
        """Save the model's weights."""
        self.model.save_weights(filename)

    def load(self, filename):
        """Load the model's weights."""
        self.model.load_weights(filename)


def main(episodes=5000, batch_size=32, load_model_filename=None):
    """Initialize a DQNAgent and play some episodes."""
    env = gym.make('deal-or-no-deal-v0')

    # FIXED AMOUNT
    state_size = 28
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)

    if load_model_filename:
        agent.load(load_model_filename)
        print('Agent loaded!')

    done = False

    for e in range(episodes):
        state = env.reset()

        state = np.append(state[0][0], [state[0][1] / 500000, state[0][2] / 10])
        state = np.reshape(state, [1, state_size]).astype(np.float32)
        for time in range(10):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)

            next_state = np.append(next_state[0],
                                   [next_state[0][1] / 500000,
                                    next_state[2] / 10])
            next_state = np.reshape(next_state, [1, state_size]).astype(np.float32)
            agent.memorize(state, action, reward, next_state, done)

            state = next_state
            if done:
                print('Episode: {}/{}, Game Length: {}, Score: {}, e: {:.3}'
                      .format(e, episodes, time + 1, reward, agent.epsilon))
                break

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        if e % 500 == 0 and e > 0:
            agent.save(f'/deal_or_no_deal/data/dqn_v2_{e}.h5')


if __name__ == '__main__':
    fire.Fire(main)
