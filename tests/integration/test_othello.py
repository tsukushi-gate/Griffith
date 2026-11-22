"""4×4オセロの手番、探索の価値、自己対戦からのゲート更新を検証する。"""

import numpy as np
import torch

from experiments.evaluate_othello_exact import solve_positions
from experiments.train_othello import (
    INITIAL,
    PASS,
    PolicyValueNet,
    legal_moves,
    outcome,
    search,
    self_play,
    train_batch,
)


def test_opening_and_multiple_flip_directions():
    moves = legal_moves(INITIAL)
    assert set(moves) == {1, 4, 11, 14}
    assert moves[1] == (
        0, -1, 0, 0,
        0, -1, -1, 0,
        0, -1, 1, 0,
        0, 0, 0, 0,
    )
    board = (0, -1, 1, 0, -1, -1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0)
    child = legal_moves(board)[0]
    assert all(child[i] == -1 for i in (0, 1, 4, 5))


def test_forced_pass_and_early_terminal():
    board = (1, -1, 0, 0) + (0,) * 12
    passed = tuple(-cell for cell in board)
    assert legal_moves(passed) == {PASS: board}
    final = legal_moves(board)[2]
    assert legal_moves(final) == {}
    assert outcome(final) == -1
    assert legal_moves((1, -1) * 8) == {}
    assert outcome((1, -1) * 8) == 0


def test_search_prefers_forced_win_to_forced_loss():
    model = PolicyValueNet()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    board = (-1, -1, -1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 0, 0)
    policy = search(model, board, 64)
    assert policy.argmax() == 15
    assert policy[14] > 0
    assert policy[:14].sum() + policy[PASS] == 0
    assert policy.sum() == 1


def test_self_play_targets_and_lut_update():
    torch.manual_seed(0)
    model = PolicyValueNet()
    rng = np.random.default_rng(2)
    samples = self_play(model, 4, rng)
    assert samples[0].board == INITIAL
    for index, sample in enumerate(samples):
        moves = legal_moves(sample.board)
        visited = np.flatnonzero(sample.policy)
        assert set(visited) <= set(moves)
        assert np.isclose(sample.policy.sum(), 1.0)
        if index + 1 < len(samples):
            following = samples[index + 1]
            assert any(moves[action] == following.board for action in visited)
            assert sample.result == -following.result
        else:
            assert any(not legal_moves(moves[a]) and sample.result == -outcome(moves[a])
                       for a in visited)
    initial_weights = model.gates.weights.detach().clone()
    losses = train_batch(model, torch.optim.Adam(model.parameters()), samples)
    assert np.isfinite(losses).all()
    assert not torch.equal(model.gates.weights, initial_weights)


def test_exact_opening_and_forced_pass_values():
    margins = solve_positions()
    assert len(margins) == 58533
    assert margins[INITIAL] == -8
    assert {margins[child] for child in legal_moves(INITIAL).values()} == {8}
    passed_board = next(board for board in margins if set(legal_moves(board)) == {PASS}
                        and margins[board] != 0)
    assert margins[passed_board] == -margins[legal_moves(passed_board)[PASS]]
