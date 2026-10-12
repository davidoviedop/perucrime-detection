"""
2_compress.py — PASO 2: poda y cuantización de un modelo entrenado, y tabla de comparación.

Para cada combinación (poda × tipo de dato) hace una copia comprimida del modelo y mide:
  - calidad: AUC macro y binario (Normal vs resto) en test, por ventana y por video (MIL)
  - costo: tamaño (MB), GFLOPs por ventana, y velocidad (FPS) en el dispositivo de
    evaluación y también en CPU para poder comparar todas las variantes en igualdad de condiciones

y escribe una fila por variante en results/comparison.csv (reemplazando las filas anteriores
del mismo run_id) y el detalle en {run_dir}/compression_sweep.csv.

  poda      global, no estructurada, por magnitud L1 sobre todos los Conv2d/Linear; sin ajuste fino
  fp32      sin cambios (la fila 0 % / fp32 reutiliza la evaluación con AMP, como el script original)
  fp16      todo el modelo en media precisión
  int8      cuantización dinámica de backbone y cabeza (en CPU; el transformer temporal queda en fp32)

Sirve para cualquier corrida entrenada: el profesor de la etapa 2 o el estudiante destilado.

Uso
---
    python 2_compress.py --run-dir outputs/<id_estudiante>
    python 2_compress.py --run-dir ../2-modelling/outputs/<id_profesor> --skip-cpu-fps
    python 2_compress.py --run-dir outputs/<id> --prune-amounts 0 0.2 --dtypes fp32 int8 --wandb
"""
import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # raíz de 3-optimization/ (agrega también 2-modelling/)

from optimization import STAGE_DIR   # al importar `optimization` queda lib/ de la etapa 2 en sys.path
import numpy as np
import torch

from lib.cli import add_override_args, add_tracking_args, parse_overrides, resolve_data_root
from lib.dataset import build_loaders
from lib.evaluation import score_predictions
from lib.model import build_model, count_parameters
from lib.runs import load_run_info
from lib.tracking import Tracker
from lib.training import evaluate
from optimization.benchmark import (measure_fps, measure_fps_cpu, measure_gflops, serialized_size_mb)
from optimization.comparison import (COMPARISON_COLUMNS, SWEEP_COLUMNS, comparison_rows,
                                     write_comparison_csv, write_sweep_csv)
from optimization.compress import cast_dtype, predict, prune_model

DEFAULT_COMPARISON_CSV = STAGE_DIR / "results" / "comparison.csv"


def main():
    parser = argparse.ArgumentParser(description="Poda y cuantización de un modelo entrenado")
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Carpeta de la corrida entrenada (con best.pt y run.yaml)")
    parser.add_argument("--comparison-csv", type=Path, default=DEFAULT_COMPARISON_CSV,
                        help="CSV de comparación compartido entre corridas (por defecto: results/comparison.csv)")
    parser.add_argument("--prune-amounts", type=float, nargs="+", default=[0.0, 0.1, 0.2],
                        help="Fracciones de pesos a podar (por defecto: 0 0.1 0.2)")
    parser.add_argument("--dtypes", nargs="+", default=["fp32", "fp16", "int8"], choices=["fp32", "fp16", "int8"],
                        help="Tipos de dato a probar (por defecto: fp32 fp16 int8)")
    parser.add_argument("--skip-cpu-fps", action="store_true",
                        help="No mide FPS en CPU (es lo más lento, sobre todo con ConvNeXt)")
    parser.add_argument("--save-variants", action="store_true",
                        help="Guarda cada variante completa (torch.save del módulo) en {run_dir}/variants/")
    parser.add_argument("--verbose", action="store_true",
                        help="Imprime la esparsidad por capa y las capas cuantizadas, y registra las métricas completas de cada variante")
    add_override_args(parser)
    add_tracking_args(parser)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    run_id = run_dir.name
    cfg, classes, sources = load_run_info(run_dir)
    for key, value in parse_overrides(args).items():
        if not hasattr(cfg, key):
            raise SystemExit(f"campo desconocido en --set: {key}")
        setattr(cfg, key, tuple(value) if isinstance(value, list) else value)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    amp = device == 'cuda'
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    normal_idx = cls_to_idx.get('Normal', -1)
    img_size = tuple(cfg.img_size)

    data_root = resolve_data_root(cfg.data_root)
    _, _, test_loader, _, _, test_recs = build_loaders(
        data_root, sources, classes, cls_to_idx,
        cfg.seq_len, cfg.stride, img_size, cfg.batch_size, cfg.num_workers,
        max_windows_per_video=cfg.max_windows_per_video, frame_stride=cfg.frame_stride)

    base_model = build_model(cfg, len(classes), pretrained=False).to(device)
    base_model.load_state_dict(torch.load(run_dir / 'best.pt', map_location=device, weights_only=True))
    base_model.eval()
    n_params = count_parameters(base_model)[0]
    print(f'Run {run_id}: {cfg.backbone}, {n_params:,} parámetros')

    try:
        gflops = measure_gflops(base_model, device, cfg.seq_len, img_size)
    except Exception as exc:
        print(f'GFLOPs measurement failed ({type(exc).__name__}: {exc}) — reporting NaN.')
        gflops = float('nan')
    print(f'GFLOPs per {cfg.seq_len}-frame window: {gflops:.2f}')

    # Evaluación del modelo completo, igual que en 2-modelling/2_evaluate.py (con AMP en CUDA)
    base_preds = evaluate(base_model, test_loader, device, len(classes), amp=amp)

    tracker = Tracker(run_dir, classes, use_wandb=args.wandb, project=cfg.wandb_project,
                      name=cfg.wandb_run_name, run_id=run_id)
    results = []
    for prune_amt in args.prune_amounts:
        for dtype in args.dtypes:
            label = f'prune{prune_amt:.0%}_{dtype}'

            # La combinación (0 % podado, fp32) es exactamente el modelo ya evaluado: se reutiliza.
            if prune_amt == 0.0 and dtype == 'fp32':
                probs, preds, labels = base_preds
                sparsity, size_mb = 0.0, serialized_size_mb(base_model)
                fps_model, fps_device = base_model, device
            else:
                print(f'\n--- {label} ---')
                m = copy.deepcopy(base_model)
                m, sparsity, prune_rows = prune_model(m, prune_amt)
                m, eval_device, quant_info = cast_dtype(m, dtype, device)

                if args.verbose:
                    if prune_rows:
                        print(f'Global sparsity: {sparsity:.1%}')
                    if quant_info.get('quantized'):
                        ql, fl = quant_info['quantized'], quant_info['fp32_linears']
                        print(f'Quantized {len(ql)}/{len(fl)} nn.Linear layers to int8:')
                        for name, shape in ql.items():
                            print(f'  {name:<40} {shape}')
                        missed = sorted(set(fl) - set(ql))
                        if missed:
                            print(f'  NOT quantized (unexpected): {missed}')

                size_mb = serialized_size_mb(m)
                if args.save_variants:
                    (run_dir / 'variants').mkdir(exist_ok=True)
                    torch.save(m, run_dir / 'variants' / f'variant_{label}.pt')
                probs, preds, labels = predict(m, test_loader, dtype, eval_device)
                fps_model, fps_device = m, eval_device

            fps = measure_fps(fps_model, dtype, fps_device, cfg.seq_len, img_size)
            fps_cpu = float('nan') if args.skip_cpu_fps else measure_fps_cpu(
                fps_model, dtype, fps_device, cfg.seq_len, img_size)

            res = score_predictions(probs, preds, labels, test_recs, classes, normal_idx, cfg,
                                    split='test', tag=label)
            f = res['flat']
            wp, vp = res['window']['prefix'], res['video']['prefix']
            results.append({
                'label': label, 'prune_amount': prune_amt, 'dtype': dtype,
                'size_mb': size_mb, 'sparsity': sparsity,
                'window_acc': res['window']['accuracy'], 'window_auc': f[wp + 'auc_macro'],
                'window_f1': f[wp + 'f1_macro'], 'window_auc_binary': f[wp + 'auc_binary'],
                'video_acc': res['video']['accuracy'], 'video_auc': f[vp + 'auc_macro'],
                'video_f1': f[vp + 'f1_macro'], 'video_auc_binary': f[vp + 'auc_binary'],
                'gflops': gflops, 'fps': fps, 'fps_cpu': fps_cpu, 'fps_device': str(fps_device),
            })
            if args.verbose:
                tracker.log(f, res['plots'])

            fps_model = None
            if device == 'cuda':
                torch.cuda.empty_cache()

    # ── Tabla final ─────────────────────────────────────────────────────────
    print(f'\n{"Config":<20} {"size_mb":>9} {"sparsity":>9} {"win_acc":>9} {"win_auc":>9} '
          f'{"vid_acc":>9} {"vid_auc":>9} {"FPS":>9} {"FPS_cpu":>9}')
    for r in results:
        print(f'{r["label"]:<20} {r["size_mb"]:>9.2f} {r["sparsity"]:>9.1%} {r["window_acc"]:>9.4f} '
              f'{r["window_auc"]:>9.4f} {r["video_acc"]:>9.4f} {r["video_auc"]:>9.4f} '
              f'{r["fps"]:>9.1f} {r["fps_cpu"]:>9.1f}')

    rows = comparison_rows(run_id, cfg.backbone, n_params, results)
    write_comparison_csv(args.comparison_csv, rows, run_id)
    write_sweep_csv(run_dir / 'compression_sweep.csv', results)
    print(f'\nComparison table written to {args.comparison_csv} ({len(rows)} rows for run {run_id})')
    print(f'Detalle por variante en {run_dir / "compression_sweep.csv"}')

    if tracker.wandb is not None:
        wb = tracker.wandb
        tracker.log({
            'compression_sweep_table': wb.Table(columns=SWEEP_COLUMNS, data=[
                [r['label'], r['prune_amount'], r['dtype'], r['size_mb'], r['sparsity'], r['window_acc'],
                 r['window_auc'], r['window_f1'], r['video_acc'], r['video_auc'], r['video_f1']] for r in results]),
            'comparison_table': wb.Table(columns=COMPARISON_COLUMNS,
                                         data=[[row[c] for c in COMPARISON_COLUMNS] for row in rows]),
        })
    tracker.finish()


if __name__ == "__main__":
    main()
