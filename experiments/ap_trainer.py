"""後方互換シム。実体は griffith_logic/training/rules/ap.py に移動した。"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from griffith_logic.training.rules.ap import APTrainer, LayerCache

__all__ = ['APTrainer', 'LayerCache']
