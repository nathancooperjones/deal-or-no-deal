import gym
from gym import spaces
from gym.utils import seeding


class Deal_or_No_Deal(gym.Env):
    """OpenAI `gym` environment for Deal or No Deal."""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        """TODO."""
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(26),
            spaces.Discrete(10),
        ))
        self._seed()

        # Start the first game
        self.reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

        return [seed]

    def step(self, action):
        """TODO."""
        pass

    def reset(self):
        """TODO."""
        pass

    def render(self, mode='human'):
        """TODO."""
        pass

    def close(self):
        """TODO."""
        pass
