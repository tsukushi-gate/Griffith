"""4×4オセロの自己対戦記録から、Griffithの方策・価値ネットワークを学習する。"""

import argparse
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from griffith_logic import LutLayer

SIZE = 4
PASS = SIZE * SIZE
ACTIONS = PASS + 1
Board = tuple[int, ...]
INITIAL: Board = (
    0, 0, 0, 0,
    0, -1, 1, 0,
    0, 1, -1, 0,
    0, 0, 0, 0,
)
DIRECTIONS = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx]


def placements(board: Board) -> dict[int, Board]:
    """置けるマスごとに、着手後の盤面を次の手番から見た向きで返す。"""
    moves = {}
    for action, stone in enumerate(board):
        if stone:
            continue
        row, col = divmod(action, SIZE)
        flips = []
        for dy, dx in DIRECTIONS:
            y, x = row + dy, col + dx
            line = []
            while 0 <= y < SIZE and 0 <= x < SIZE:
                index = y * SIZE + x
                if board[index] != -1:
                    if board[index] == 1:
                        flips.extend(line)
                    break
                line.append(index)
                y, x = y + dy, x + dx
        if flips:
            following = [-cell for cell in board]
            following[action] = -1
            for index in flips:
                following[index] = -1
            moves[action] = tuple(following)
    return moves


def legal_moves(board: Board) -> dict[int, Board]:
    """片方だけが置けない局面ではパス、双方が置けない終局では空の辞書を返す。"""
    moves = placements(board)
    if moves:
        return moves
    opponent = tuple(-cell for cell in board)
    return {PASS: opponent} if placements(opponent) else {}


def outcome(board: Board) -> int:
    difference = sum(board)
    return (difference > 0) - (difference < 0)


class PolicyValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.gates = LutLayer(2 * PASS, 256)
        self.policy = nn.Linear(256, ACTIONS)
        self.value = nn.Linear(256, 1)

    def forward(self, board: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        planes = torch.cat((board == 1, board == -1), dim=-1).float()
        features = self.gates(planes) - 0.5
        return self.policy(features), self.value(features).squeeze(-1).tanh()


@dataclass
class Node:
    board: Board
    prior: float = 1.0
    visits: int = 0
    value_sum: float = 0.0
    children: dict[int, 'Node'] = field(default_factory=dict)

    def expand(self, model: PolicyValueNet) -> float:
        moves = legal_moves(self.board)
        if not moves:
            return outcome(self.board)
        logits, value = model(torch.tensor([self.board]))
        priors = logits[0, list(moves)].softmax(dim=0).tolist()
        self.children = {
            action: Node(board, prior)
            for (action, board), prior in zip(moves.items(), priors, strict=True)
        }
        return value.item()

    def select(self) -> 'Node':
        exploration = 1.5 * math.sqrt(max(1, self.visits))

        def score(child: Node) -> float:
            # 子の価値は相手の手番から見た値なので、親が手を選ぶときは符号を反転する。
            value = -child.value_sum / child.visits if child.visits else 0.0
            return value + exploration * child.prior / (1 + child.visits)

        return max(self.children.values(), key=score)


@torch.no_grad()
def search(model: PolicyValueNet, board: Board, simulations: int,
           rng: np.random.Generator | None = None) -> np.ndarray:
    """非終局の合法手に対する訪問割合を返す。自己対戦時のみ根に探索ノイズを加える。"""
    if simulations < 1:
        raise ValueError('simulations must be positive')
    root = Node(board)
    root.expand(model)
    if not root.children:
        raise ValueError('cannot search a terminal board')
    if rng is not None:
        noise = rng.dirichlet(np.full(len(root.children), 0.3))
        for child, sample in zip(root.children.values(), noise, strict=True):
            child.prior = 0.75 * child.prior + 0.25 * sample

    for _ in range(simulations):
        node = root
        path = [root]
        while node.children:
            node = node.select()
            path.append(node)
        value = node.expand(model)
        for visited in reversed(path):
            visited.visits += 1
            visited.value_sum += value
            value = -value

    policy = np.zeros(ACTIONS)
    for action, child in root.children.items():
        policy[action] = child.visits / simulations
    return policy


@dataclass
class Sample:
    board: Board
    policy: np.ndarray
    result: int


def self_play(model: PolicyValueNet, simulations: int, rng: np.random.Generator) -> list[Sample]:
    board = INITIAL
    history = []
    while moves := legal_moves(board):
        policy = search(model, board, simulations, rng)
        history.append((board, policy))
        board = moves[rng.choice(ACTIONS, p=policy)]

    result = outcome(board)
    samples = []
    for position, policy in reversed(history):
        result = -result
        samples.append(Sample(position, policy, result))
    return samples[::-1]


def train_batch(model: PolicyValueNet, optimizer: torch.optim.Optimizer,
                batch: list[Sample]) -> tuple[float, float]:
    boards = torch.tensor([sample.board for sample in batch])
    policies = torch.from_numpy(np.stack([sample.policy for sample in batch])).float()
    results = torch.tensor([sample.result for sample in batch], dtype=torch.float32)
    logits, values = model(boards)
    policy_loss = F.cross_entropy(logits, policies)
    value_loss = F.mse_loss(values, results)
    optimizer.zero_grad()
    (policy_loss + value_loss).backward()
    optimizer.step()
    return policy_loss.item(), value_loss.item()


def positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iterations', type=positive_int, default=5)
    parser.add_argument('--games', type=positive_int, default=8, help='self-play games per iteration')
    parser.add_argument('--simulations', type=positive_int, default=32, help='MCTS simulations per move')
    parser.add_argument('--updates', type=positive_int, default=32, help='gradient steps per iteration')
    parser.add_argument('--batch-size', type=positive_int, default=64)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', type=Path, default=Path('checkpoints/othello4.pt'))
    parser.add_argument('--snapshot-iteration', type=positive_int,
                        help='also save the model after this iteration, beside --output')
    args = parser.parse_args()
    if args.snapshot_iteration is not None and args.snapshot_iteration > args.iterations:
        parser.error('--snapshot-iteration must be at most --iterations')

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    # eval() は Griffith のゲートを離散化するため、自己対戦も学習と同じソフトゲートで行う。
    model = PolicyValueNet().train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    replay: deque[Sample] = deque(maxlen=8192)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for iteration in range(1, args.iterations + 1):
        positions = 0
        for _ in range(args.games):
            game = self_play(model, args.simulations, rng)
            replay.extend(game)
            positions += len(game)
        pool = list(replay)
        losses = []
        for _ in range(args.updates):
            indices = rng.choice(len(pool), size=min(len(pool), args.batch_size), replace=False)
            losses.append(train_batch(model, optimizer, [pool[i] for i in indices]))
        policy_loss, value_loss = np.mean(losses, axis=0)
        print(f'iteration={iteration} games={args.games} positions={positions} replay={len(pool)} '
              f'policy_loss={policy_loss:.4f} value_loss={value_loss:.4f}', flush=True)
        if iteration == args.snapshot_iteration:
            snapshot = args.output.with_stem(f'{args.output.stem}_iter_{iteration}')
            torch.save(model.state_dict(), snapshot)

    torch.save(model.state_dict(), args.output)
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
