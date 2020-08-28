from gym.envs.registration import register

from ._version import __version__


register(
    id='deal-or-no-deal-v0',
    entry_point='deal_or_no_deal.envs:Deal_or_No_Deal',
)
