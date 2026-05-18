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
import sys, os, subprocess

if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    try: sys.stdout.reconfigure(encoding='utf-8')
    except: pass

sys.path.insert(0, os.path.dirname(__file__))
BASE = os.path.dirname(os.path.abspath(__file__))

def p(rel: str) -> str:
    return os.path.join(BASE, rel)

# ── 自动安装依赖 ──────────────────────────────────────────────────────────────

def _ensure_reqs() -> None:
    """Auto-install packages listed in requirements.txt."""
    req_path = p('requirements.txt')
    if not os.path.exists(req_path):
        return
    with open(req_path, 'r') as f:
        pkgs = [l.strip() for l in f if l.strip() and not l.startswith('#')]
    for pkg in pkgs:
        try:
            __import__(pkg.split('>=')[0].split('=')[0].split('<')[0].strip())
        except ImportError:
            print(f"[setup] Installing {pkg} ...")
            subprocess.check_call(
                [sys.executable, '-m', 'pip', 'install', pkg, '-q'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

_ensure_reqs()

# ── 命令分发 ──────────────────────────────────────────────────────────────────

def main() -> None:
    args = sys.argv[1:]
    cmd = args[0] if args else 'help'

    if cmd == 'run':
        from scripts.core import AtomicCrystalGrowth, _make_table
        text = open(p('corpus/example.txt'), encoding='utf-8').read()
        g = AtomicCrystalGrowth(text)
        r = g.run()
        print(f"\nSample segmentation (first 15 sentences):")
        tbl = [[str(i+1),
                ' / '.join(s[:15]) + (' …' if len(s) > 15 else '')]
               for i, s in enumerate(r[:15])]
        print(_make_table(tbl, headers=['#', 'Segmentation'],
                          colalign=('center', 'left')))

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
