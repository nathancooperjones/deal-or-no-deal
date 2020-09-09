import numpy as np
import pandas as pd

from deal_or_no_deal.config import CASE_VALUE_TO_MAPPING


def preprocess_historical_case_data(df, keep_contestant_id=False):
    """
    Format the Deal or No Deal TV dataset into a model-ready dataset.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame of Deal or No Deal TV spreadsheet
    keep_contestant_id: bool
        Keep the `'Contestant ID'` column or not (default False)

    Returns
    -------
    case_data_df: pd.DataFrame
        DataFrame with columns:
            * `case_X`: a binary flag of whether the case has been opened or not in that round
            * `round_got_better`: a binary flag indiciating if the worst case opened in the round
              was the final case (`round_got_better` = False) or not
            * `round`: round number in the game [1, 9]
            * `expected_value`: EV of the cases left in the game
            * `offer`: banker's offer made at the end of that round
            * `percentage_difference`: percentage difference the offer is from the expected value

    """
    df_data = df[
        [
            'Contestant ID',
            'Data',
            'Unnamed: 4',
            'Unnamed: 5',
            'Unnamed: 6',
            'Unnamed: 7',
            'Unnamed: 8',
            'Unnamed: 9',
        ]
    ]

    (
        cases_opened_list,
        banker_offer_list,
        round_num_list,
        expected_value_list,
        round_got_better_list,
        contestant_id_list,
    ) = _condense_df_to_lists(df_data)

    case_data_df = pd.DataFrame(
        cases_opened_list,
        columns=['case_' + str(key) for key in CASE_VALUE_TO_MAPPING.keys()],
    )

    case_data_df['round_got_better'] = round_got_better_list
    case_data_df['round'] = round_num_list
    case_data_df['expected_value'] = expected_value_list
    case_data_df['offer'] = banker_offer_list

    if keep_contestant_id:
        case_data_df['contestant_id'] = contestant_id_list

    case_data_df['percentage_difference'] = (
        case_data_df['offer'] / case_data_df['expected_value']
    )

    return case_data_df


def _condense_df_to_lists(df_data):
    # get our data in a format we can actually use
    cases_opened_list = list()
    banker_offer_list = list()
    round_num_list = list()
    expected_value_list = list()
    round_got_better_list = list()
    contestant_id_list = list()

    # start at low numbers we will immediatently overwrite
    contestant_id_temp = 0
    cases_opened_temp_list = None
    round_num = 0

    for idx, row in df_data.iterrows():
        # get rid of NaNs
        row = row[~pd.isnull(row)].values

        # separate out the contestant ID, cases, and banker offer
        contestant_id = row[0]
        cases_opened = row[1:-1]
        banker_offer = row[-1]

        if 1 + len(cases_opened) + 1 != len(row):
            raise ValueError('Data format is not as expected!')

        # figure out if this is a continuation of a game or a new one
        if contestant_id != contestant_id_temp:
            # we are in a new game
            contestant_id_temp = contestant_id
            cases_opened_temp_list = cases_opened.tolist()
            round_num = 1
        else:
            # we are continuing a game
            cases_opened = cases_opened_temp_list + cases_opened.tolist()
            cases_opened_temp_list = cases_opened
            round_num += 1

        # get rid of any rows where the banker didn't make an offer
        if 'None' in row:
            continue

        if len(cases_opened) > 1:
            round_got_better = int(cases_opened[-1] != max(cases_opened))

        cases_opened_binary = [
            int(key in cases_opened) for key in CASE_VALUE_TO_MAPPING.keys()
        ]

        cases_opened_list.append(cases_opened_binary)
        banker_offer_list.append(banker_offer)
        round_num_list.append(round_num)
        expected_value_list.append(_calculate_expected_value(cases_opened_binary))
        round_got_better_list.append(round_got_better)
        contestant_id_list.append(contestant_id)

    return (
        cases_opened_list,
        banker_offer_list,
        round_num_list,
        expected_value_list,
        round_got_better_list,
        contestant_id_list,
    )


def _calculate_expected_value(case_list):
    """Calculate expected value of unopened cases from binary string of case game state."""
    case_value_to_mapping_array = np.array(list(CASE_VALUE_TO_MAPPING.keys()))

    return case_value_to_mapping_array[np.array(case_list) == 0].mean()
