import argparse
import json
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


def solve_positions() -> dict[Board, int]:
    margins = {}

    def solve(board: Board) -> int:
        if board not in margins:
            moves = legal_moves(board)
            margins[board] = max((-solve(child) for child in moves.values()), default=sum(board))
        return margins[board]

    solve(INITIAL)
    return margins


def outcome_name(margin: int) -> str:
    if margin > 0:
        return 'win'
    return 'draw' if margin == 0 else 'loss'


@torch.no_grad()
def evaluate(model: PolicyValueNet, margins: dict[Board, int], simulations: int) -> dict:
    all_moves = Counter()
    consequential = Counter()
    terminal = 0
    policy_preserved = 0
    for board, margin in margins.items():
        moves = legal_moves(board)
        if not moves:
            terminal += 1
            continue
        outcomes = {action: outcome_name(-margins[child]) for action, child in moves.items()}
        best = outcome_name(margin)
        if len(set(outcomes.values())) == 1:
            chosen = best
        else:
            actions = list(moves)
            logits, _ = model(torch.tensor([board]))
            policy_action = actions[int(logits[0, actions].argmax())]
            policy_preserved += outcomes[policy_action] == best
            action = int(search(model, board, simulations).argmax())
            chosen = outcomes[action]
            consequential[f'{best}_to_{chosen}'] += 1
        all_moves[f'{best}_to_{chosen}'] += 1
    preserved = consequential['win_to_win'] + consequential['draw_to_draw']
    return {
        'outcome_preserved_fraction': preserved / consequential.total(),
        'policy_outcome_preserved_fraction': policy_preserved / consequential.total(),
        'states': len(margins), 'terminal': terminal, 'initial_black_margin': margins[INITIAL],
        'all_nonterminal': dict(all_moves), 'outcome_depends_on_move': dict(consequential),
    }


def main():
    parser = argparse.ArgumentParser(description='Compare a checkpoint with exact 4x4 Othello outcomes')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--checkpoint', type=Path)
    source.add_argument('--untrained-seed', type=int, help='evaluate the initial model for this training seed')
    parser.add_argument('--simulations', type=positive_int, default=32)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.untrained_seed is not None:
        torch.manual_seed(args.untrained_seed)
    model = PolicyValueNet().train()
    if args.checkpoint is not None:
        model.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=True))
    result = evaluate(model, solve_positions(), args.simulations)
    origin = ({'checkpoint': str(args.checkpoint)} if args.checkpoint is not None
              else {'untrained_seed': args.untrained_seed})
    print(json.dumps({**origin, 'simulations': args.simulations, **result}, indent=2))


if __name__ == '__main__':
    main()
