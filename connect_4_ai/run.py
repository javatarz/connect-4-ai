from typing import List, Tuple

import torch
from kaggle_environments import evaluate

from connect_4_ai.model_nn import create_model
from connect_4_ai.torch_utils import device


def _get_win_def(player_number: int) -> List[int]:
    if player_number == 1:
        return [1, -1]
    else:
        return [-1, 1]


def _get_invalid_def(player_number: int) -> List[int]:
    if player_number == 1:
        return [None, 0]
    else:
        return [0, None]


def _get_agent_stats(player_number: int, outcomes: List[List[int]],
                     agent_only_outcomes: List[List[int]]) -> Tuple[float, float, float]:
    win_rate = outcomes.count(_get_win_def(player_number)) / len(outcomes)
    win_rate_when_first = agent_only_outcomes.count(_get_win_def(player_number)) / len(agent_only_outcomes)
    invalid_move_rate = outcomes.count(_get_invalid_def(player_number)) / len(outcomes)

    return win_rate, win_rate_when_first, invalid_move_rate


def _print_agent_stats(player_number: int, outcomes: List[List[int]], agent_only_outcomes: List[List[int]]) -> None:
    win_rate, win_rate_when_first, invalid_move_rate = _get_agent_stats(player_number, outcomes, agent_only_outcomes)
    print(f"Agent {player_number} stats\n{win_rate=} {win_rate_when_first=} {invalid_move_rate=}")


def get_win_percentages(agent1, agent2, n_rounds=100):
    # Use default Connect Four setup
    config = {'rows': 6, 'columns': 7, 'inarow': 4}
    # Agent 1 goes first (roughly) half the time
    agent_1_round_count = n_rounds // 2
    agent_1_first = evaluate("connectx", [agent1, agent2], config, [], agent_1_round_count)
    # Agent 2 goes first (roughly) half the time
    agent_2_round_count = n_rounds - agent_1_round_count
    agent_2_first = [[b, a] for [a, b] in evaluate("connectx", [agent2, agent1], config, [], agent_2_round_count)]

    outcomes = agent_1_first + agent_2_first

    _print_agent_stats(1, outcomes, agent_1_first)
    print()
    _print_agent_stats(2, outcomes, agent_2_first)


def run() -> None:
    torch.set_default_device(device())
    model = create_model()
    model.load("ppo_connect4")
    get_win_percentages(agent1=model, agent2=)


if __name__ == "__main__":
    run()
