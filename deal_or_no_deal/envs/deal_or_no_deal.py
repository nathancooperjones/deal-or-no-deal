import random

import gym
from gym import spaces
from gym.utils import seeding
import joblib
import numpy as np

from deal_or_no_deal.config import CASES


class Deal_or_No_Deal(gym.Env):
    """OpenAI `gym` environment for Deal or No Deal."""
    def __init__(self, verbose=False):
        """TODO."""
        # two actions: No Deal or Deal
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Tuple((
            # which cases have been opened
            spaces.Box(np.zeros(26), np.ones(26), dtype=np.float32),
            # bankers offer
            spaces.Box(np.array([0]), np.array([1_000_000]), dtype=np.float32),
            # round number
            spaces.Discrete(9),
        ))
        self._seed()

        self.trials = 100
        self.verbose = verbose

        self.banker_model = joblib.load('/deal_or_no_deal/data/banker_model_0906.pkl')

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

        return [seed]

    def step(self, action):
        """TODO."""
        assert self.action_space.contains(action)

        if action == 1:
            self.final_amount = self.banker_offers[-1]

            while self.round_num < 9:
                self._open_cases_and_make_offer()

            self.game_done = True

        return self._open_cases_and_make_offer()

    def reset(self):
        """Reset all game stats and start the first round."""
        self.cases_left = CASES.copy()
        self.player_case = self.cases_left.pop(random.randrange(len(self.cases_left)))
        self.round_num = 0

        self.banker_offers = list()
        self.final_amount = -1
        self.game_done = False

        self.cases_opened = list()

        return self._open_cases_and_make_offer()

    def _open_cases_and_make_offer(self):
        """Open the number of cases necessary for this round, and make an offer."""
        self.round_num += 1

        # check for end of game
        if self.round_num >= 9 or self.game_done:
            if self.final_amount == -1:
                self.final_amount = self.player_case

            banker_offer = 0
            reward = self._check_if_win()

        else:
            number_of_cases_to_open = max(1, 7 - self.round_num)

            self.cases_opened = list()
            for _ in range(number_of_cases_to_open):
                case_opened = self.cases_left.pop(random.randrange(len(self.cases_left)))
                self.cases_opened.append(case_opened)
                if self.verbose:
                    print(f'Opened case {case_opened}')

            banker_offer = self._make_banker_offer()
            reward = 0

        return self._get_observation_to_return(banker_offer), reward, self.game_done, {}

    def _get_observation_to_return(self, banker_offer):
        return (
            np.array([0 if case in self.cases_left else 1 for case in CASES]),
            np.array([banker_offer]),
            np.array([self.round_num]),
        )

    def _check_if_win(self):
        """Determine whether we have won."""
        self.game_done = True
        self.cases_left = list()

        if self.final_amount >= max(self.banker_offers):
            return 1
        else:
            return -1

    def _make_banker_offer(self):
        """Make an offer from the Banker."""
        # TODO: make model prediction
        round_got_better = 0
        if len(self.cases_opened) > 1:
            round_got_better = int(self.cases_opened[-1] != max(self.cases_opened))

        cases_opened_binary = [int(key not in self.cases_left) for key in CASES]
        round = self.round_num
        expected_value = np.mean(self.cases_left)

        model_input = cases_opened_binary + [round_got_better] + [round] + [expected_value]
        model_input = np.array([model_input])

        banker_offer = expected_value * self.banker_model.predict(model_input)

        self.banker_offers.append(banker_offer)

        return banker_offer

    def render(self, mode='human'):
        """Render the environment for a human."""
        print('Cases Left:  ', self.cases_left)
        print('Round Number:', self.round_num)
