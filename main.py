"""
Atomic Crystal Growth — Self-Organizing Unsupervised Word Segmentation

Usage:
    python main.py [command]

Commands:
    run      Run segmentation on corpus/example.txt
    eval     Compare with jieba baseline
    train    Leave-one-year-out cross-validation on corpus/corpus.json
"""
from __future__ import annotations
import sys, os

if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    try: sys.stdout.reconfigure(encoding='utf-8')
    except: pass

sys.path.insert(0, os.path.dirname(__file__))
BASE = os.path.dirname(os.path.abspath(__file__))


def p(rel: str) -> str:
    return os.path.join(BASE, rel)


def _check_deps() -> None:
    """Check that required packages are installed; print friendly message if not."""
    missing = []
    for mod, pip_name in [('tabulate', 'tabulate'), ('jieba', 'jieba')]:
        try:
            __import__(mod)
        except ImportError:
            missing.append(pip_name)
    if missing:
        print(f"[setup] Missing dependencies: {', '.join(missing)}")
        print(f"[setup] Run: pip install {' '.join(missing)}")
        sys.exit(1)


_check_deps()


# ── Distribution ──────────────────────────────────────────────────────────────────

def main() -> None:
    args = sys.argv[1:]
    cmd = args[0] if args else 'help'

    if cmd == 'run':
        from scripts.core import AtomicCrystalGrowth, _make_table
        text = open(p('corpus/example.txt'), encoding='utf-8').read()
        g = AtomicCrystalGrowth(text)
        r = g.run()
        print(f"\nSample segmentation:")
        tbl = [[str(i + 1), ' / '.join(s[:15]) + (' ...' if len(s) > 15 else '')] for i, s in enumerate(r[:15])]
        print(_make_table(tbl, headers=['#', 'Segmentation'], colalign=('center', 'left')))

    elif cmd == 'eval':
        path = args[1] if len(args) > 1 else p('corpus/example.txt')
        from scripts.evaluation import compare
        compare(path)

    elif cmd == 'train':
        path = args[1] if len(args) > 1 else p('corpus/corpus.json')
        from scripts.train import main as train_main
        train_main(path)

    else:
        print(__doc__.strip())


if __name__ == '__main__':
    main()
