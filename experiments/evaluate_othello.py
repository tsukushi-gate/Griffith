"""4×4オセロの学習前後のモデルを、同じ開始局面から先後交代で対戦させる。"""

import argparse
from collections import Counter
from pathlib import Path

import torch

from experiments.train_othello import (
    INITIAL,
    Board,
    PolicyValueNet,
    legal_moves,
    positive_int,
    search,
)


def openings_after_four_plies() -> list[Board]:
    positions = {INITIAL}
    for _ in range(4):
        positions = {child for board in positions for child in legal_moves(board).values()}
    return sorted(board for board in positions if legal_moves(board))


def final_black_margin(board: Board, black: PolicyValueNet, white: PolicyValueNet,
                       simulations: int) -> int:
    black_to_move = True
    while moves := legal_moves(board):
        model = black if black_to_move else white
        policy = search(model, board, simulations)
        board = moves[int(policy.argmax())]
        black_to_move = not black_to_move
    return sum(board) if black_to_move else -sum(board)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('checkpoints/othello4.pt'))
    parser.add_argument('--opponent', type=Path, help='same-seed opponent checkpoint (default: untrained)')
    parser.add_argument('--seed', type=int, default=0, help='original training seed')
    parser.add_argument('--simulations', type=positive_int, default=32)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    initial = PolicyValueNet().train()
    trained = PolicyValueNet().train()
    trained.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=True))
    if not torch.equal(initial.gates.indices, trained.gates.indices):
        parser.error('initial wiring differs from checkpoint; check the training seed')
    if args.opponent is not None:
        initial.load_state_dict(torch.load(args.opponent, map_location='cpu', weights_only=True))
        if not torch.equal(initial.gates.indices, trained.gates.indices):
            parser.error('opponent wiring differs from checkpoint; use the same training seed')

    for name, boards in [('standard', [INITIAL]), ('four_plies', openings_after_four_plies())]:
        for color, black, white, sign in [('black', trained, initial, 1), ('white', initial, trained, -1)]:
            counts = Counter()
            margin_sum = 0
            for board in boards:
                margin = sign * final_black_margin(board, black, white, args.simulations)
                counts[(margin > 0) - (margin < 0)] += 1
                margin_sum += margin
            print(f'{name} trained={color} W/D/L={counts[1]}/{counts[0]}/{counts[-1]} '
                  f'games={len(boards)} mean_disc_margin={margin_sum / len(boards):+.2f}', flush=True)


if __name__ == '__main__':
    main()
