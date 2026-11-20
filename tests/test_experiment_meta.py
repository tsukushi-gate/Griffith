"""実験メタデータ規律の契約テスト。

- init_wandb のシグネチャ後方互換(notes等 未指定で従来動作)
- normalize_tags の形式
- 実験台帳(experiments-ledger.md)追記関数の単体
- finishレコードがjsonlへ確実に書かれる(WANDB_MODE=offline で実走)

wandbへの実通信は行わない(offlineモード or 純関数)。
"""

import json
import os
import sys

import pytest

# experiments/ を import path に追加(wandb_utils を読む)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))

import wandb_utils  # noqa: E402

HAS_WANDB = wandb_utils.HAS_WANDB


# ============================================================
# normalize_tags
# ============================================================

class TestNormalizeTags:
    def test_all_fields(self):
        tags = wandb_utils.normalize_tags(
            model='logic_gpt', dataset='shakespeare', seed=0, series='D1',
        )
        # 順序は series → model → dataset → seed
        assert tags == [
            'series:D1', 'model:logic_gpt', 'dataset:shakespeare', 'seed:0',
        ]

    def test_none_fields_are_skipped(self):
        assert wandb_utils.normalize_tags(model='cnn') == ['model:cnn']
        assert wandb_utils.normalize_tags() == []

    def test_seed_zero_included(self):
        # seed=0 は falsy だが含めなければならない
        assert 'seed:0' in wandb_utils.normalize_tags(seed=0)


# ============================================================
# add_experiment_args
# ============================================================

class TestAddExperimentArgs:
    def test_adds_notes_and_hypothesis(self):
        import argparse

        parser = argparse.ArgumentParser()
        wandb_utils.add_experiment_args(parser)
        args = parser.parse_args([])
        assert args.notes is None
        assert args.hypothesis is None

        args2 = parser.parse_args(['--notes', 'x', '--hypothesis', 'y'])
        assert args2.notes == 'x'
        assert args2.hypothesis == 'y'


# ============================================================
# 実験台帳の追記
# ============================================================

class TestLedger:
    def test_pick_metric_priority(self):
        # val_loss が最優先
        assert wandb_utils._pick_ledger_metric(
            {'val_loss': 1.5, 'loss': 2.0}
        ) == 'val_loss=1.5000'
        # val_loss 無ければ best_val_loss
        assert wandb_utils._pick_ledger_metric(
            {'best_val_loss': 0.3, 'loss': 2.0}
        ) == 'best_val_loss=0.3000'
        # どれも無ければ '-'
        assert wandb_utils._pick_ledger_metric({'foo': 1.0}) == '-'

    def test_append_creates_header_and_row(self, tmp_path):
        ledger = tmp_path / 'docs' / 'experiments-ledger.md'
        wandb_utils.append_ledger_row(
            str(ledger), date='2026-07-10', run_id='abc123',
            group='logic_gpt', job_type='train',
            hypothesis='仮説X', notes='変更Y', final_metric='val_loss=1.2345',
        )
        text = ledger.read_text(encoding='utf-8')
        # ヘッダが生成されている
        assert '# 実験台帳' in text
        assert '| 日付 | run ID | group |' in text
        # 行が追記され、判定列は空欄
        assert '| 2026-07-10 | abc123 | logic_gpt | train | 仮説X | 変更Y '\
               '| val_loss=1.2345 |  |' in text

    def test_append_twice_keeps_single_header(self, tmp_path):
        ledger = tmp_path / 'experiments-ledger.md'
        for rid in ('r1', 'r2'):
            wandb_utils.append_ledger_row(
                str(ledger), date='2026-07-10', run_id=rid, group='g',
                job_type='train', hypothesis='', notes='',
                final_metric='-',
            )
        text = ledger.read_text(encoding='utf-8')
        assert text.count('# 実験台帳') == 1
        assert 'r1' in text and 'r2' in text

    def test_escape_pipe_in_cell(self):
        row = wandb_utils.format_ledger_row(
            '2026-07-10', 'id', 'g', 'train',
            hypothesis='a|b', notes='c\nd', final_metric='-',
        )
        # パイプはエスケープ、改行は空白化
        assert 'a\\|b' in row
        assert 'c d' in row


# ============================================================
# should_use_wandb(前提: offlineは有効)
# ============================================================

class TestShouldUseWandb:
    @pytest.mark.skipif(not HAS_WANDB, reason='wandb未インストール')
    def test_offline_enabled_without_api_key(self, monkeypatch):
        monkeypatch.delenv('WANDB_API_KEY', raising=False)
        monkeypatch.setenv('WANDB_MODE', 'offline')
        assert wandb_utils.should_use_wandb() is True

    @pytest.mark.skipif(not HAS_WANDB, reason='wandb未インストール')
    def test_disabled_mode(self, monkeypatch):
        monkeypatch.setenv('WANDB_MODE', 'disabled')
        assert wandb_utils.should_use_wandb() is False


# ============================================================
# init_wandb 後方互換 + finishレコード(offline実走)
# ============================================================

def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest.mark.skipif(not HAS_WANDB, reason='wandb未インストール')
class TestInitAndFinishOffline:
    @pytest.fixture(autouse=True)
    def _offline_env(self, tmp_path, monkeypatch):
        # offlineモードでtmp cwdへ隔離(jsonl/台帳/wandbディレクトリが散らからない)
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv('WANDB_API_KEY', raising=False)
        monkeypatch.setenv('WANDB_MODE', 'offline')
        monkeypatch.setenv('WANDB_SILENT', 'true')
        monkeypatch.setenv('WANDB_PROJECT', 'griffith-logic-test')
        # finishガードをテスト間でリセット
        wandb_utils._FINISHED_RUN_IDS.clear()
        self.tmp_path = tmp_path
        yield
        # 後片付け: runが残っていれば閉じる
        import wandb
        if wandb.run is not None:
            wandb.finish()

    def test_backward_compat_no_meta(self):
        """notes/hypothesis 未指定でも従来どおり init レコードが書かれる。"""
        ok = wandb_utils.init_wandb('smoke', {'lr': 0.01})
        assert ok is True
        entries = _read_jsonl(self.tmp_path / '.wandb_runs.jsonl')
        init_rec = entries[-1]
        assert init_rec['run_name'].startswith('smoke_')
        assert init_rec['notes'] is None
        assert init_rec['hypothesis'] is None
        assert init_rec['job_type'] == 'train'

    def test_notes_hypothesis_recorded(self):
        import wandb

        wandb_utils.init_wandb(
            'smoke', {'lr': 0.01}, notes='何を変えたか', hypothesis='仮説Z',
            tags=['t1'] + wandb_utils.normalize_tags(model='m', seed=0),
        )
        # config に experiment/hypothesis が入る
        assert wandb.run.config['experiment/hypothesis'] == '仮説Z'
        # wandb notes は「notes + git=」の結合(gitが取れなくても notes は残る)
        assert '何を変えたか' in (wandb.run.notes or '')
        # jsonl init レコード
        init_rec = _read_jsonl(self.tmp_path / '.wandb_runs.jsonl')[-1]
        assert init_rec['notes'] == '何を変えたか'
        assert init_rec['hypothesis'] == '仮説Z'

    def test_finish_writes_record_and_ledger(self):
        import wandb

        # 台帳は既存ファイルにのみ追記する(pip版にはdocs/が無いため)
        ledger_file = self.tmp_path / 'docs' / 'experiments-ledger.md'
        ledger_file.parent.mkdir(parents=True, exist_ok=True)
        ledger_file.write_text(wandb_utils.LEDGER_HEADER, encoding='utf-8')

        wandb_utils.init_wandb(
            'smoke', {'lr': 0.01}, group='logic_gpt',
            notes='変更', hypothesis='仮説',
        )
        run_id = wandb.run.id
        wandb.log({'loss': 0.5, 'best_val_loss': 0.3})
        wandb_utils.finish_wandb()

        # jsonl に finish レコード(final_metrics は数値のみ)
        entries = _read_jsonl(self.tmp_path / '.wandb_runs.jsonl')
        finish_recs = [e for e in entries if e.get('event') == 'finish']
        assert len(finish_recs) == 1
        fr = finish_recs[0]
        assert fr['run_id'] == run_id
        assert fr['status'] == 'finished'
        assert 'finished_at' in fr
        assert fr['final_metrics'].get('best_val_loss') == pytest.approx(0.3)
        assert fr['final_metrics'].get('loss') == pytest.approx(0.5)

        # 台帳に1行(final は優先度で best_val_loss を採用)
        ledger = self.tmp_path / 'docs' / 'experiments-ledger.md'
        assert ledger.is_file()
        ltext = ledger.read_text(encoding='utf-8')
        assert run_id in ltext
        assert 'best_val_loss=0.3000' in ltext

    def test_finish_idempotent(self):
        """finish を二重に呼んでも finish レコード・台帳行は1つだけ。"""
        import wandb

        wandb_utils.init_wandb('smoke', {'lr': 0.01}, group='g')
        run_id = wandb.run.id
        wandb_utils.finish_wandb()
        wandb_utils.finish_wandb()  # 二重呼び出し(atexit相当)

        entries = _read_jsonl(self.tmp_path / '.wandb_runs.jsonl')
        finish_recs = [
            e for e in entries
            if e.get('event') == 'finish' and e.get('run_id') == run_id
        ]
        assert len(finish_recs) == 1
