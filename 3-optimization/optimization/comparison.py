"""
optimization/comparison.py — tabla de comparación (comparison.csv) de variantes de modelos.

Una fila por (modelo, poda, tipo de dato), con las columnas de la tesis. El CSV se comparte
entre corridas: al repetir una corrida se reemplazan solo sus filas (mismo run_id).
"""
import csv
import os
from pathlib import Path

COMPARISON_COLUMNS = [
    'run_id', 'backbone', 'quantization_applied', 'prune_pct', 'num_parameters',
    'size_mb', 'GFLOPS', 'FPS', 'FPS_device', 'FPS_cpu',
    'test_window_auc_macro', 'test_window_auc_binary',
    'test_video_auc_macro', 'test_video_auc_binary',
]

# Columnas del detalle por variante que se guarda junto a la corrida (compression_sweep.csv)
SWEEP_COLUMNS = ['config', 'prune_amount', 'dtype', 'size_mb', 'sparsity',
                 'window_acc', 'window_auc', 'window_f1', 'video_acc', 'video_auc', 'video_f1']


def comparison_rows(run_id, backbone, num_parameters, results):
    """Filas de comparison.csv a partir de los resultados del barrido.

    num_parameters es el conteo de parámetros de la arquitectura (igual en todas las
    variantes): la poda solo pone ceros e int8 empaqueta sus pesos fuera de .parameters(),
    así que un conteo "vivo" por variante engañaría.
    """
    return [{
        'run_id': run_id, 'backbone': backbone,
        'quantization_applied': 'none' if r['dtype'] == 'fp32' else r['dtype'],
        'prune_pct': round(r['prune_amount'] * 100, 2), 'num_parameters': num_parameters,
        'size_mb': round(r['size_mb'], 3), 'GFLOPS': round(r['gflops'], 3), 'FPS': round(r['fps'], 2),
        'FPS_device': r['fps_device'], 'FPS_cpu': round(r['fps_cpu'], 2),
        'test_window_auc_macro':  round(r['window_auc'], 4),
        'test_window_auc_binary': round(r['window_auc_binary'], 4),
        'test_video_auc_macro':   round(r['video_auc'], 4),
        'test_video_auc_binary':  round(r['video_auc_binary'], 4),
    } for r in results]


def write_comparison_csv(path, rows, run_id):
    """Agrega `rows` a comparison.csv reemplazando las filas anteriores del mismo run_id."""
    path = Path(path)
    kept = []
    if path.exists() and path.stat().st_size > 0:
        with open(path, newline='', encoding='utf-8') as f:
            kept = [r for r in csv.DictReader(f) if r.get('run_id') != run_id]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=COMPARISON_COLUMNS, lineterminator='\n')
        writer.writeheader()
        writer.writerows(kept)
        writer.writerows(rows)
    os.replace(tmp, path)


def write_sweep_csv(path, results):
    """Detalle de la corrida (todas las variantes): exactitud, F1, esparsidad, tamaño."""
    path = Path(path)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(SWEEP_COLUMNS)
        for r in results:
            writer.writerow([r['label'], r['prune_amount'], r['dtype'], r['size_mb'], r['sparsity'],
                             r['window_acc'], r['window_auc'], r['window_f1'],
                             r['video_acc'], r['video_auc'], r['video_f1']])
