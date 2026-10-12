"""
Asignación de items a splits: búsqueda local con doble objetivo.

Objetivo (menor es mejor), sumado sobre cada (clase, split):
  ((ventanas_en_split - ventanas_objetivo) / ventanas_objetivo)^2   balance de ventanas
+ ((items_en_split   - items_objetivo)   / items_objetivo)^2         balance de items

El término de items importa porque algunas clases concentran la mayoría de sus
ventanas en unos pocos items "ballena". Balancear solo las ventanas dejaría
felizmente una ballena entera en un solo split; el término de items obliga a la
búsqueda a repartir también la diversidad entre splits.

Búsqueda: construcción voraz -> movimientos de un item -> intercambios de pares,
repetido desde varias semillas. El orden de construcción alterna entre ordenado
por tamaño y aleatorio, porque el orden aleatorio escapa de óptimos locales en
los que el ordenado por tamaño se queda atascado (no quitar la alternancia al
afinar).
"""
import random
from collections import defaultdict

from .config import SPLIT_NAMES, SPLIT_RATIOS


def score(window_counts, item_counts, target_windows, target_items, classes):
    total = 0.0
    for c in classes:
        for s in SPLIT_NAMES:
            tw = target_windows[c][s]
            if tw > 0:
                total += ((window_counts[s].get(c, 0) - tw) / tw) ** 2
            ti = target_items[c][s]
            if ti > 0:
                total += ((item_counts[s].get(c, 0) - ti) / ti) ** 2
    return total


def assign_splits(item_class_windows, classes, seed, order_mode):
    n_windows_total = defaultdict(int)
    n_items_total   = defaultdict(int)
    for classes_of_item in item_class_windows.values():
        for c, w in classes_of_item.items():
            n_windows_total[c] += w
            n_items_total[c]   += 1

    target_windows = {c: {s: n_windows_total[c] * SPLIT_RATIOS[s] for s in SPLIT_NAMES} for c in classes}
    target_items   = {c: {s: n_items_total[c]   * SPLIT_RATIOS[s] for s in SPLIT_NAMES} for c in classes}

    rng = random.Random(seed)
    item_ids = sorted(item_class_windows)   # orden determinista: el resultado no depende del orden de las filas del CSV
    rng.shuffle(item_ids)
    if order_mode == "size":
        item_ids.sort(key=lambda iid: -sum(item_class_windows[iid].values()))

    window_counts = {s: defaultdict(int) for s in SPLIT_NAMES}
    item_counts   = {s: defaultdict(int) for s in SPLIT_NAMES}
    assignment    = {}

    def add(iid, split):
        for c, w in item_class_windows[iid].items():
            window_counts[split][c] += w
            item_counts[split][c]   += 1

    def remove(iid, split):
        for c, w in item_class_windows[iid].items():
            window_counts[split][c] -= w
            item_counts[split][c]   -= 1

    def cur_score():
        return score(window_counts, item_counts, target_windows, target_items, classes)

    # Construcción voraz
    for iid in item_ids:
        best_s, best_sc = None, None
        for s in SPLIT_NAMES:
            add(iid, s)
            sc = cur_score()
            remove(iid, s)
            if best_sc is None or sc < best_sc:
                best_sc, best_s = sc, s
        add(iid, best_s)
        assignment[iid] = best_s

    # Fase 1: movimientos de un solo item
    improved, passes = True, 0
    while improved and passes < 50:
        improved = False
        passes += 1
        for iid in item_ids:
            cur = assignment[iid]
            remove(iid, cur)
            best_s, best_sc = cur, None
            for s in SPLIT_NAMES:
                add(iid, s)
                sc = cur_score()
                remove(iid, s)
                if best_sc is None or sc < best_sc - 1e-9:
                    best_sc, best_s = sc, s
            add(iid, best_s)
            if best_s != cur:
                assignment[iid] = best_s
                improved = True

    # Fase 2: intercambios de pares de items entre splits distintos
    improved, passes = True, 0
    while improved and passes < 30:
        improved = False
        passes += 1
        for i in range(len(item_ids)):
            for j in range(i + 1, len(item_ids)):
                a, b = item_ids[i], item_ids[j]
                sa, sb = assignment[a], assignment[b]
                if sa == sb:
                    continue
                before = cur_score()
                remove(a, sa); remove(b, sb)
                add(a, sb); add(b, sa)
                after = cur_score()
                if after < before - 1e-9:
                    assignment[a], assignment[b] = sb, sa
                    improved = True
                else:
                    remove(a, sb); remove(b, sa)
                    add(a, sa); add(b, sb)

    return assignment, window_counts, item_counts, cur_score()


def best_assignment(item_class_windows, classes, n_restarts):
    best = None
    for seed in range(n_restarts):
        order_mode = "size" if seed % 3 == 0 else "random"
        assignment, window_counts, item_counts, sc = assign_splits(
            item_class_windows, classes, seed, order_mode)
        if best is None or sc < best[0]:
            best = (sc, assignment, window_counts, item_counts)
    return best
