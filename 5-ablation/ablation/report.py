"""
ablation/report.py — junta las corridas de una suite en una tabla (CSV y Markdown).

Por variante: media ± desviación entre semillas de cada métrica de test y la diferencia (Δ)
con la referencia. Con una sola semilla no hay desviación y los Δ son solo indicativos: el
conjunto de test es pequeño (58 videos; 4 de Robbery y 4 de Shooting), así que diferencias
de uno o dos puntos están dentro del ruido.
"""
import csv
import json
from pathlib import Path

import numpy as np

METRICS = [
    ('test_video_auc_macro', 'AUC macro (video)'),
    ('test_video_f1_macro', 'F1 macro (video)'),
    ('test_video_auc_binary', 'AUC binario (video)'),
    ('test_video_accuracy', 'Exactitud (video)'),
    ('test_window_auc_macro', 'AUC macro (ventana)'),
    ('test_window_f1_macro', 'F1 macro (ventana)'),
    ('val_window_auc_macro', 'AUC macro val (ventana)'),
]


def collect(suite_dir: Path):
    rows = []
    for d in sorted(Path(suite_dir).iterdir()):
        if not (d / 'metrics_test.json').exists() or not (d / 'variant.json').exists():
            continue
        info = json.loads((d / 'variant.json').read_text())
        flat = json.loads((d / 'metrics_test.json').read_text())['metrics']
        if (d / 'metrics_val.json').exists():
            flat.update(json.loads((d / 'metrics_val.json').read_text())['metrics'])
        rows.append({'variant': info['variant'], 'seed': info['seed'], 'info': info, 'm': flat,
                     'classes': json.loads((d / 'metrics_test.json').read_text())['classes']})
    return rows


def aggregate(rows, variant_order, reference):
    out = []
    ref_means = {}
    for name in variant_order:
        rs = [r for r in rows if r['variant'] == name]
        if not rs:
            continue
        agg = {'variant': name, 'n_seeds': len(rs),
               'params_total_M': rs[0]['info'].get('params_total', float('nan')) / 1e6,
               'params_trainable_M': rs[0]['info'].get('params_trainable', float('nan')) / 1e6,
               'epochs_run': float(np.mean([r['info'].get('epochs_run', np.nan) for r in rs]))}
        for key, _ in METRICS:
            vals = np.array([r['m'].get(key, np.nan) for r in rs], dtype=float)
            agg[key] = float(np.nanmean(vals))
            agg[key + '_std'] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else float('nan')
        out.append(agg)
        if name == reference:
            ref_means = {k: agg[k] for k, _ in METRICS}
    for agg in out:
        for key, _ in METRICS:
            agg[key + '_delta'] = agg[key] - ref_means.get(key, float('nan'))
    return out


def per_class_recall(rows, variant_order):
    classes = rows[0]['classes'] if rows else []
    table = []
    for name in variant_order:
        rs = [r for r in rows if r['variant'] == name]
        if not rs:
            continue
        table.append({'variant': name, **{c: float(np.nanmean([r['m'].get(f'test_video_recall_{c}', np.nan) for r in rs]))
                                          for c in classes}})
    return classes, table


def _fmt(mean, std):
    if mean != mean:
        return '—'
    return f'{mean:.3f}' + (f' ± {std:.3f}' if std == std else '')


def write(suite_name, agg, classes, recall, results_dir: Path, reference):
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f'ablation_{suite_name}.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        cols = ['variant', 'n_seeds', 'params_total_M', 'params_trainable_M', 'epochs_run'] + \
               [c for k, _ in METRICS for c in (k, k + '_std', k + '_delta')]
        w = csv.DictWriter(f, fieldnames=cols, lineterminator='\n')
        w.writeheader()
        w.writerows(agg)

    head = ['Variante', 'Sem.', 'Parám. (M)'] + [lbl for _, lbl in METRICS[:6]] + [f'Δ {METRICS[0][1]}', f'Δ {METRICS[1][1]}']
    lines = [f'# Ablación `{suite_name}` (test)', '',
             f'Referencia: `{reference}`. Media ± desviación entre semillas (sin desviación si hay una sola). '
             'Test pequeño (58 videos): diferencias de 1–2 puntos están dentro del ruido.', '',
             '| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    for a in agg:
        cells = [f"`{a['variant']}`", str(a['n_seeds']), f"{a['params_total_M']:.2f}"]
        cells += [_fmt(a[k], a[k + '_std']) for k, _ in METRICS[:6]]
        cells += [f"{a[METRICS[0][0] + '_delta']:+.3f}", f"{a[METRICS[1][0] + '_delta']:+.3f}"]
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines += ['', '## Recall por clase (video)', '',
              '| Variante | ' + ' | '.join(classes) + ' |', '|' + '---|' * (len(classes) + 1)]
    for r in recall:
        lines.append(f"| `{r['variant']}` | " + ' | '.join(f'{r[c]:.2f}' for c in classes) + ' |')
    md_path = results_dir / f'ablation_{suite_name}.md'
    md_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return csv_path, md_path
