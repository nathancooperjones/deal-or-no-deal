import random

import joblib
import numpy as np

from deal_or_no_deal.config import CASES


class Deal_or_No_Deal_Fast_Play():
    """
    Mock environment for predicting future game states of Deal or No Deal based on a current one.
    While the `__init__` simply loads in the Banker model, `generate_future_game_states` handles the
    logic for determining whether we should continue playing the game or not.

    """
    def __init__(self, banker_model_filename):
        """Initialize the environment."""
        self.banker_model = joblib.load(banker_model_filename)

    def generate_future_game_states(self,
                                    cases_opened,
                                    round_num,
                                    offer,
                                    number_of_games_to_run=100):
        """
        Given a current game state, play the game `number_of_games_to_run` times onward to determine
        the percentage of the time you end up with winnings greater than the current `offer`.

        Parameters
        ----------
        cases_opened: iterable, length 26
            A binary array indicating whether a case has been opened or not
        round_num: int
            Round number of the current game
        offer: float
            Offer the banker has just made
        number_of_games_to_run: int
            Number of games to run in the future to generate `probability_we_should_continue`. Note
            that the number will be more stable as it increases, but it will take more time to
            compute (default 100)

        Returns
        -------
        probability_we_should_continue: float
            A percentage of future games that end in winnings higher than `offer`

        """
        should_we_continue_list = list()

        while len(should_we_continue_list) < number_of_games_to_run:
            self.round_num = round_num + 1
            self.cases_left = [CASES[idx] for idx in range(len(cases_opened)) if cases_opened[idx] == 0]
            self.player_case = self.cases_left.pop(random.randrange(len(self.cases_left)))

            should_we_continue_for_this_game = False

            while self.round_num <= 9:
                if offer < self._open_cases_and_make_offer():
                    should_we_continue_for_this_game = True
                    break

                self.round_num += 1

            should_we_continue_list.append(should_we_continue_for_this_game)

        probability_we_should_continue = should_we_continue_list.count(True) / len(should_we_continue_list)

        return probability_we_should_continue

    def _open_cases_and_make_offer(self):
        """Open the number of cases necessary for this round, and make an offer."""
        # check for end of game
        if self.round_num >= 9:
            return self.player_case

        else:
            number_of_cases_to_open = max(1, 7 - self.round_num)

            cases_opened = list()
            for _ in range(number_of_cases_to_open):
                cases_opened.append(self.cases_left.pop(random.randrange(len(self.cases_left))))

            banker_offer = self._make_banker_offer(cases_opened)

            return banker_offer

    def _make_banker_offer(self, cases_opened):
        """Make an offer from the Banker."""
        round_got_better = 0
        if len(cases_opened) > 1:
            round_got_better = int(cases_opened[-1] != max(cases_opened))

        cases_opened_binary = [int(key not in self.cases_left) for key in CASES]
        round = self.round_num
        expected_value = np.mean(self.cases_left)

        model_input = cases_opened_binary + [round_got_better] + [round] + [expected_value]
        model_input = np.array([model_input])

        banker_offer = expected_value * self.banker_model.predict(model_input)

        return banker_offer[0]
