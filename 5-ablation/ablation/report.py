"""
ablation/report.py — junta las corridas de una suite en las tablas del informe de la ablación.

Para cada variante: media ± desviación estándar entre semillas de cada métrica de test; Δ frente a la
referencia con IC 95 % por bootstrap pareado por video (sobre las probabilidades promediadas entre
semillas, ver stats.py); veredicto «aporta / tendencia / no aporta»; y las métricas de validación para
comprobar que el ranking no depende solo del test.
"""
import csv
import datetime as dt
import json
import shutil
from pathlib import Path

import numpy as np

from .stats import SeedAveragedPredictions, paired_delta_ci, stratified_video_draws, verdict


def archive(paths, results_dir: Path):
    """Copia cada archivo de resultados a results/history/{nombre}_{fecha-hora}{ext}, para que las tablas de una
    ejecución no se pierdan cuando la siguiente sobrescribe la versión «actual» (results/{nombre})."""
    history = Path(results_dir) / 'history'
    history.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
    out = []
    for p in paths:
        p = Path(p)
        dest = history / f'{p.stem}_{stamp}{p.suffix}'
        shutil.copy2(p, dest)
        out.append(dest)
    return out


def collect(suite_dir: Path):
    """{id: [ {seed, info, test, val, classes, dir} ]} con las corridas que tienen variant.json."""
    runs = {}
    for d in sorted(Path(suite_dir).iterdir()):
        if not (d / 'variant.json').exists():
            continue
        info = json.loads((d / 'variant.json').read_text())
        rec = {'seed': info['seed'], 'info': info, 'dir': d}
        for split in ('val', 'test'):
            f = d / f'metrics_{split}.json'
            if f.exists():
                j = json.loads(f.read_text())
                rec[split] = j['metrics']
                rec['classes'] = j['classes']
        runs.setdefault(info['id'], []).append(rec)
    return runs


def _vals(rs, split, key):
    return np.array([r[split].get(key, np.nan) for r in rs if split in r], dtype=float)


def mean_sd(rs, split, key):
    v = _vals(rs, split, key)
    if not len(v) or np.all(np.isnan(v)):
        return float('nan'), float('nan')
    return float(np.nanmean(v)), (float(np.nanstd(v, ddof=1)) if len(v) > 1 else float('nan'))


def fmt(m, sd, nd=3):
    if m != m:
        return '—'
    return f'{m:.{nd}f}' + (f' ± {sd:.{nd}f}' if sd == sd else '')


def class_counts(rs):
    """Videos y ventanas de test por clase (de la primera semilla de la referencia)."""
    z = np.load(rs[0]['dir'] / 'preds_test.npz')
    classes = rs[0]['classes']
    labels, vpaths = z['labels'], z['vpaths']
    out = []
    for i, c in enumerate(classes):
        m = labels == i
        out.append((c, len(set(vpaths[m].tolist())), int(m.sum())))
    return out


def analyze(runs, order, reference, n_boot=2000, window_ci=True, seed=0):
    """Una fila por variante con todas las cifras del informe."""
    classes = runs[reference][0]['classes']
    ref_pred = SeedAveragedPredictions([r['dir'] / 'preds_test.npz' for r in runs[reference]])
    draws = stratified_video_draws(ref_pred.video_labels, n_boot, seed)
    rows = []
    for vid in order:
        rs = runs.get(vid)
        if not rs:
            continue
        row = {'id': vid, 'question': rs[0]['info'].get('question', ''), 'n_seeds': len(rs),
               'params_M': rs[0]['info']['params_total'] / 1e6, 'gflops_window': rs[0]['info'].get('gflops_window', np.nan),
               'train_seconds': float(np.nanmean([r['info'].get('train_seconds', np.nan) for r in rs])),
               'epochs_run': float(np.mean([r['info'].get('epochs_run', np.nan) for r in rs])),
               'best_epoch': float(np.mean([r['info'].get('best_epoch', np.nan) for r in rs]))}
        for split in ('test', 'val'):
            for lvl in ('video', 'window'):
                for k in ('auc_macro', 'auc_binary', 'f1_macro', 'accuracy'):
                    key = f'{split}_{lvl}_{k}'
                    row[key], row[key + '_sd'] = mean_sd(rs, split, key)
        for c in classes:
            key = f'test_video_auc_{c}'
            row[key], row[key + '_sd'] = mean_sd(rs, 'test', key)
        if vid != reference:
            var_paths = [r['dir'] / 'preds_test.npz' for r in rs]
            pred = SeedAveragedPredictions(var_paths)
            ref_use, var_use, draws_use = ref_pred, pred, draws
            ref_sd = ref_row_value(rows, reference, 'test_video_auc_macro_sd')
            var_sd = row['test_video_auc_macro_sd']
            ref_mean = ref_row_value(rows, reference, 'test_video_auc_macro')
            var_mean = row['test_video_auc_macro']
            if pred.videos != ref_pred.videos:
                # La variante no puede puntuar todos los videos (p. ej. A5: los clips más cortos que una ventana quedan sin
                # ventanas). Δ, IC y desviaciones se calculan sobre los videos comunes, restringiendo también la referencia.
                common = sorted(set(ref_pred.videos) & set(pred.videos))
                ref_paths = [r['dir'] / 'preds_test.npz' for r in runs[reference]]
                ref_use = SeedAveragedPredictions(ref_paths, videos=common)
                var_use = SeedAveragedPredictions(var_paths, videos=common)
                draws_use = stratified_video_draws(ref_use.video_labels, n_boot, seed)
                per_seed_ref = [SeedAveragedPredictions([p], videos=common).video_auc() for p in ref_paths]
                per_seed_var = [SeedAveragedPredictions([p], videos=common).video_auc() for p in var_paths]
                ref_mean, var_mean = float(np.mean(per_seed_ref)), float(np.mean(per_seed_var))
                ref_sd, var_sd = float(np.std(per_seed_ref, ddof=1)), float(np.std(per_seed_var, ddof=1))
                row.update({'common_videos': len(common), 'variant_videos': len(pred.videos), 'reference_videos': len(ref_pred.videos),
                            'common_ref_auc_macro_video': ref_mean, 'common_var_auc_macro_video': var_mean})
            d, lo, hi = paired_delta_ci(ref_use.video_auc, var_use.video_auc, draws_use)
            row.update({'delta_video_auc': d, 'ci_lo': lo, 'ci_hi': hi})
            if window_ci:
                dw, lw, hw = paired_delta_ci(ref_use.window_auc, var_use.window_auc, draws_use)
                row.update({'delta_window_auc': dw, 'ci_window_lo': lw, 'ci_window_hi': hw})
            # Δ medio entre semillas (el que compara la regla de decisión con la desviación entre semillas)
            seed_delta = var_mean - ref_mean
            row['seed_delta_video_auc'] = seed_delta
            row['verdict'] = verdict(seed_delta, ref_sd, var_sd, lo, hi)
            row['direction'] = '↑' if seed_delta > 0 else '↓'
        else:
            row['verdict'] = 'referencia'
        rows.append(row)
    return rows, classes


def ref_row_value(rows, reference, key):
    return next(r for r in rows if r['id'] == reference)[key]


def spearman(rows, key_a, key_b):
    from scipy.stats import spearmanr
    a = [r[key_a] for r in rows]
    b = [r[key_b] for r in rows]
    return float(spearmanr(a, b).correlation) if len(rows) > 2 else float('nan')


def write(suite_name, rows, classes, counts, results_dir: Path, reference, n_boot):
    results_dir.mkdir(parents=True, exist_ok=True)
    cols = sorted({k for r in rows for k in r})
    with (results_dir / f'ablation_{suite_name}.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator='\n')
        w.writeheader()
        w.writerows(rows)

    L = [f'# Ablación de modelado `{suite_name}`', '',
         f'Referencia: `{reference}`. Métricas de **test**, media ± desviación estándar entre semillas. Δ = variante − referencia '
         f'con IC 95 % por bootstrap pareado por video ({n_boot} remuestreos estratificados por clase, sobre las probabilidades '
         'promediadas entre semillas). Regla: «aporta» si |Δ medio| supera la desviación entre semillas (la mayor de las dos) '
         'Y el IC excluye 0; «tendencia» si se cumple solo una; «no aporta» si ninguna.', '',
         '## Test: videos y ventanas por clase', '', '| Clase | Videos | Ventanas |', '|---|---|---|']
    L += [f'| {c} | {v} | {w_} |' for c, v, w_ in counts]
    L += ['', '## Tabla principal', '',
          '| ID | Pregunta | AUC-Macro video | Δ vs ' + reference + ' [IC 95 %] | AUC-Macro ventana | AUC Shooting | Parám. (M) | GFLOPs/ventana | Veredicto |',
          '|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        delta = '—' if r['id'] == reference else f"{r['delta_video_auc']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]"
        verd = r['verdict'] + (f" {r['direction']}" if r['id'] != reference and r['verdict'] != 'no aporta' else '')
        L.append(f"| {r['id']} | {r['question']} | {fmt(r['test_video_auc_macro'], r['test_video_auc_macro_sd'])} | {delta} | "
                 f"{fmt(r['test_window_auc_macro'], r['test_window_auc_macro_sd'])} | "
                 f"{fmt(r.get('test_video_auc_Shooting', np.nan), r.get('test_video_auc_Shooting_sd', np.nan))} | "
                 f"{r['params_M']:.2f} | {r['gflops_window']:.1f} | {verd} |")
    restricted = [r for r in rows if 'common_videos' in r]
    if restricted:
        L += ['', '> **Videos distintos.** ' + '; '.join(
            f"{r['id']} solo puntúa {r['variant_videos']} de los {r['reference_videos']} videos de test (los clips más cortos que su ventana "
            f"no tienen ventanas); su Δ, IC y veredicto se calculan sobre los {r['common_videos']} videos comunes, restringiendo también la "
            f"referencia (AUC-Macro video en esos videos: {r['common_ref_auc_macro_video']:.3f} frente a {r['common_var_auc_macro_video']:.3f}). "
            'Sus columnas de métricas son sobre sus propios videos y no son comparables con las demás.' for r in restricted)]
    L += ['', '## Métricas secundarias (test, video)', '',
          '| ID | AUC binario | F1-Macro | Exactitud | Δ AUC-Macro ventana [IC 95 %] |', '|---|---|---|---|---|']
    for r in rows:
        dw = '—' if r['id'] == reference or 'delta_window_auc' not in r else \
            f"{r['delta_window_auc']:+.3f} [{r['ci_window_lo']:+.3f}, {r['ci_window_hi']:+.3f}]"
        L.append(f"| {r['id']} | {fmt(r['test_video_auc_binary'], r['test_video_auc_binary_sd'])} | "
                 f"{fmt(r['test_video_f1_macro'], r['test_video_f1_macro_sd'])} | "
                 f"{fmt(r['test_video_accuracy'], r['test_video_accuracy_sd'])} | {dw} |")
    L += ['', '## AUC por clase (test, video; solo descriptivo)', '',
          '| ID | ' + ' | '.join(classes) + ' |', '|' + '---|' * (len(classes) + 1)]
    for r in rows:
        L.append(f"| {r['id']} | " + ' | '.join(fmt(r.get(f'test_video_auc_{c}', np.nan), r.get(f'test_video_auc_{c}_sd', np.nan), 2)
                                                 for c in classes) + ' |')
    L += ['', '## Validación (el ranking no debe depender solo del test)', '',
          '| ID | AUC-Macro video (val) | AUC-Macro ventana (val) | Mejor época (media) | Épocas (media) | Tiempo por corrida (min) |', '|---|---|---|---|---|---|']
    for r in rows:
        L.append(f"| {r['id']} | {fmt(r['val_video_auc_macro'], r['val_video_auc_macro_sd'])} | "
                 f"{fmt(r['val_window_auc_macro'], r['val_window_auc_macro_sd'])} | {r['best_epoch']:.1f} | {r['epochs_run']:.1f} | "
                 f"{r['train_seconds'] / 60:.0f} |")
    rho = spearman(rows, 'val_video_auc_macro', 'test_video_auc_macro')
    L += ['', f'Correlación de Spearman entre el AUC-Macro por video de validación y de test, sobre las {len(rows)} configuraciones: **{rho:.2f}**.']
    path = results_dir / f'ablation_{suite_name}.md'
    path.write_text('\n'.join(L) + '\n', encoding='utf-8')
    archive([path, results_dir / f'ablation_{suite_name}.csv'], results_dir)
    return path
