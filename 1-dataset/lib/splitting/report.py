"""Reportes por consola: estadísticas del pool, la división elegida y la diversidad de items por celda."""
from collections import defaultdict

from .config import SPLIT_NAMES


def print_pool_stats(classes, item_class_windows):
    n_items_total = defaultdict(int)
    n_windows_total = defaultdict(int)
    for cw in item_class_windows.values():
        for c, w in cw.items():
            n_items_total[c] += 1
            n_windows_total[c] += w
    print(f'\n{"class":<16} {"items":>7} {"windows":>9}')
    for c in classes:
        print(f'{c:<16} {n_items_total[c]:>7} {n_windows_total[c]:>9}')


def print_report(classes, item_class_windows, window_counts, item_counts):
    n_items_total = defaultdict(int)
    for classes_of_item in item_class_windows.values():
        for c in classes_of_item:
            n_items_total[c] += 1
    print(f'\n{"class":<16} {"windows train/val/test":<24} {"items train/val/test":<22} total')
    for c in classes:
        wc = [window_counts[s].get(c, 0) for s in SPLIT_NAMES]
        ic = [item_counts[s].get(c, 0) for s in SPLIT_NAMES]
        print(f'{c:<16} {str(wc):<24} {str(ic):<22} {sum(wc)} windows / {n_items_total[c]} items')


def print_diversity(item_class_windows, assignment, classes):
    by_cs = defaultdict(lambda: defaultdict(int))
    for iid, split in assignment.items():
        for c, w in item_class_windows[iid].items():
            by_cs[(c, split)][iid] += w
    print(f'\n{"class":<16} {"split":<6} {"n_items":>7} {"windows":>8} {"max_item_share":>14}')
    for c in classes:
        for s in SPLIT_NAMES:
            d = by_cs[(c, s)]
            if not d:
                continue
            counts = sorted(d.values(), reverse=True)
            total = sum(counts)
            share = counts[0] / total if total else 0
            print(f'{c:<16} {s:<6} {len(counts):>7} {total:>8} {share:>13.0%}')
