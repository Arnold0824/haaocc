import argparse
import pickle
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = REPO_ROOT / 'work_dirs/primary_results'
DEFAULT_COMPARE_RESULTS_DIR = REPO_ROOT / 'work_dirs/reference_results'
DEFAULT_OUTPUT_DIR = REPO_ROOT / 'bev_vis'
INFO_CANDIDATES = [
    REPO_ROOT / 'data/nuscenes/bevdetv2-nuscenes_infos_val.pkl',
    REPO_ROOT / 'data/nuscenes/bevdetv2-nuscenes_infos_train.pkl',
]

POINT_CLOUD_RANGE = np.array([-40.0, -40.0, -1.0, 40.0, 40.0, 5.4],
                             dtype=np.float32)
VOXEL_SIZE = np.array([0.4, 0.4, 0.4], dtype=np.float32)
FREE_LABEL = 17

SEMANTIC_NAMES = [
    'others',
    'barrier',
    'bicycle',
    'bus',
    'car',
    'construction_vehicle',
    'motorcycle',
    'pedestrian',
    'traffic_cone',
    'trailer',
    'truck',
    'driveable_surface',
    'other_flat',
    'sidewalk',
    'terrain',
    'manmade',
    'vegetation',
]

SEMANTIC_COLORS = np.array(
    [
        [160, 160, 160],
        [112, 128, 144],
        [220, 20, 60],
        [255, 127, 80],
        [255, 158, 0],
        [233, 150, 70],
        [255, 61, 99],
        [0, 0, 230],
        [47, 79, 79],
        [255, 140, 0],
        [255, 99, 71],
        [0, 207, 191],
        [175, 0, 75],
        [75, 0, 75],
        [112, 180, 60],
        [222, 184, 135],
        [0, 175, 0],
    ],
    dtype=np.float32,
) / 255.0


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize occupancy predictions in BEV with LiDAR '
        'points overlayed in black.')
    parser.add_argument(
        '--results-dir',
        default=str(DEFAULT_RESULTS_DIR),
        help='Directory that contains scene/token/pred.npz results.')
    parser.add_argument(
        '--output-dir',
        default=str(DEFAULT_OUTPUT_DIR),
        help='Directory to save BEV overlays.')
    parser.add_argument(
        '--compare-results-dir',
        default=str(DEFAULT_COMPARE_RESULTS_DIR),
        help='Optional second results directory to render alongside the '
        'primary results.')
    parser.add_argument(
        '--sample-token',
        default=None,
        help='Only render one sample token when set.')
    parser.add_argument(
        '--max-samples',
        type=int,
        default=50,
        help='How many samples to render. Use 0 to render all matched '
        'samples.')
    parser.add_argument(
        '--pred-size',
        type=float,
        default=14.0,
        help='Marker size of flattened occupancy cells.')
    parser.add_argument(
        '--lidar-size',
        type=float,
        default=3.0,
        help='Marker size of LiDAR points.')
    parser.add_argument(
        '--dpi',
        type=int,
        default=600,
        help='DPI for saved figures.')
    return parser.parse_args()


def iter_prediction_files(results_dir: Path):
    return sorted(results_dir.rglob('pred.npz'))


def build_prediction_index(results_dir: Path):
    pred_index = {}
    if not results_dir.exists():
        return pred_index
    for pred_path in iter_prediction_files(results_dir):
        try:
            _, sample_token = load_prediction(pred_path)
        except Exception as exc:
            print(f'[skip] failed to read prediction file {pred_path}: {exc}')
            continue
        pred_index[sample_token] = pred_path
    return pred_index


def load_info_index():
    info_index = {}
    for info_path in INFO_CANDIDATES:
        if not info_path.exists():
            continue
        with open(info_path, 'rb') as f:
            data = pickle.load(f)
        infos = data['infos'] if isinstance(data, dict) and 'infos' in data else data
        for info in infos:
            token = info.get('token')
            if token and token not in info_index:
                info_index[token] = info
    if not info_index:
        raise FileNotFoundError(
            'Cannot find nuScenes info files under data/nuscenes. '
            'Expected one of: ' + ', '.join(str(p) for p in INFO_CANDIDATES))
    return info_index


def load_prediction(npz_path: Path):
    data = np.load(npz_path, allow_pickle=True)
    pred = data['pred']
    sample_token = str(data['sample_token'].tolist())
    return pred, sample_token


def resolve_lidar_path(info):
    lidar_path = Path(info['lidar_path'])
    if lidar_path.is_absolute():
        return lidar_path
    return REPO_ROOT / lidar_path


def load_lidar_points(info):
    lidar_path = resolve_lidar_path(info)
    points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)[:, :3]
    mask = (
        (points[:, 0] >= POINT_CLOUD_RANGE[0]) &
        (points[:, 0] <= POINT_CLOUD_RANGE[3]) &
        (points[:, 1] >= POINT_CLOUD_RANGE[1]) &
        (points[:, 1] <= POINT_CLOUD_RANGE[4]) &
        (points[:, 2] >= POINT_CLOUD_RANGE[2]) &
        (points[:, 2] <= POINT_CLOUD_RANGE[5])
    )
    return points[mask]


def load_gt_occ(info):
    gt_path = Path(info['occ_path']) / 'labels.npz'
    if not gt_path.is_absolute():
        gt_path = REPO_ROOT / gt_path
    gt_data = np.load(gt_path)
    return gt_data['semantics']


def flatten_occ_to_bev(pred_occ):
    occupied = pred_occ != FREE_LABEL
    z_axis = np.arange(pred_occ.shape[2], dtype=np.int16)[None, None, :]
    height_map = np.where(occupied, z_axis, -1)
    top_idx = height_map.argmax(axis=2)
    has_pred = height_map.max(axis=2) >= 0

    if not np.any(has_pred):
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int64)

    top_semantics = np.take_along_axis(
        pred_occ, top_idx[..., None], axis=2).squeeze(2)
    grid_x, grid_y = np.nonzero(has_pred)
    raw_x = POINT_CLOUD_RANGE[0] + (grid_x + 0.5) * VOXEL_SIZE[0]
    raw_y = POINT_CLOUD_RANGE[1] + (grid_y + 0.5) * VOXEL_SIZE[1]
    # Match the occupancy grid orientation to the LiDAR BEV frame.
    x = -raw_y
    y = raw_x
    labels = top_semantics[grid_x, grid_y].astype(np.int64)
    return x, y, labels


def build_legend(labels):
    handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='black',
               markeredgecolor='white', markeredgewidth=1.0, markersize=7,
               label='GT LiDAR')
    ]
    for label in sorted(set(labels.tolist())):
        if label == FREE_LABEL or label >= len(SEMANTIC_NAMES):
            continue
        handles.append(
            Patch(facecolor=SEMANTIC_COLORS[label],
                  edgecolor='none',
                  label=SEMANTIC_NAMES[label]))
    return handles


def render_sample(occ_grid, lidar_points, info, out_path: Path, pred_size: float,
                  lidar_size: float, dpi: int):
    bev_x, bev_y, labels = flatten_occ_to_bev(occ_grid)
    pred_colors = SEMANTIC_COLORS[labels] if labels.size else np.empty((0, 3))
    lidar_halo_size = max(lidar_size * 3.5, lidar_size + 6.0)

    fig, ax = plt.subplots(figsize=(9, 9), dpi=dpi)
    if bev_x.size:
        ax.scatter(
            bev_x,
            bev_y,
            c=pred_colors,
            s=pred_size,
            marker='s',
            alpha=0.80,
            linewidths=0,
            rasterized=True,
            zorder=1)
    ax.scatter(
        lidar_points[:, 0],
        lidar_points[:, 1],
        c='white',
        s=lidar_halo_size,
        alpha=0.98,
        linewidths=0,
        rasterized=True,
        zorder=3)
    ax.scatter(
        lidar_points[:, 0],
        lidar_points[:, 1],
        c='black',
        s=lidar_size,
        alpha=0.95,
        linewidths=0,
        rasterized=True,
        zorder=4)

    ax.set_xlim(float(POINT_CLOUD_RANGE[0]), float(POINT_CLOUD_RANGE[3]))
    ax.set_ylim(float(POINT_CLOUD_RANGE[1]), float(POINT_CLOUD_RANGE[4]))
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f"{info['scene_name']} | {info['token'][:8]}")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def save_legend(out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    legend_labels = np.arange(len(SEMANTIC_NAMES), dtype=np.int64)
    handles = build_legend(legend_labels)

    fig, ax = plt.subplots(figsize=(5.2, 6.0), dpi=180)
    ax.axis('off')
    ax.legend(
        handles=handles,
        loc='center',
        fontsize=10,
        framealpha=1.0,
        ncol=1,
        borderpad=0.8,
        labelspacing=0.6,
        handlelength=1.4,
        handletextpad=0.6)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    compare_results_dir = Path(args.compare_results_dir)
    output_dir = Path(args.output_dir)

    if not results_dir.exists():
        raise FileNotFoundError(f'Results directory does not exist: {results_dir}')

    info_index = load_info_index()
    pred_index = build_prediction_index(results_dir)
    if not pred_index:
        raise FileNotFoundError(f'No pred.npz files found under {results_dir}')
    compare_pred_index = build_prediction_index(compare_results_dir)
    if compare_results_dir.exists() and not compare_pred_index:
        print(f'[warn] no pred.npz files found under {compare_results_dir}')

    save_legend(output_dir / '1legend.png')

    rendered = 0
    for sample_token in sorted(pred_index.keys()):
        if args.sample_token and sample_token != args.sample_token:
            continue
        pred_path = pred_index[sample_token]
        pred_occ, sample_token = load_prediction(pred_path)
        info = info_index.get(sample_token)
        if info is None:
            print(f'[skip] missing dataset info for token {sample_token}')
            continue

        lidar_points = load_lidar_points(info)
        gt_occ = load_gt_occ(info)
        ours_path = output_dir / info['scene_name'] / f'{sample_token}_ours.png'
        gt_path = output_dir / info['scene_name'] / f'{sample_token}_gt.png'
        render_sample(
            occ_grid=pred_occ,
            lidar_points=lidar_points,
            info=info,
            out_path=ours_path,
            pred_size=args.pred_size,
            lidar_size=args.lidar_size,
            dpi=args.dpi)
        render_sample(
            occ_grid=gt_occ,
            lidar_points=lidar_points,
            info=info,
            out_path=gt_path,
            pred_size=args.pred_size,
            lidar_size=args.lidar_size,
            dpi=args.dpi)
        compare_pred_path = compare_pred_index.get(sample_token)
        if compare_pred_path is not None:
            compare_occ, _ = load_prediction(compare_pred_path)
            compare_path = output_dir / info['scene_name'] / f'{sample_token}_flashocc.png'
            render_sample(
                occ_grid=compare_occ,
                lidar_points=lidar_points,
                info=info,
                out_path=compare_path,
                pred_size=args.pred_size,
                lidar_size=args.lidar_size,
                dpi=args.dpi)
            print(f'[{rendered + 1}] saved {compare_path}')
        elif compare_results_dir.exists():
            print(f'[skip] missing flashocc result for token {sample_token}')
        rendered += 1
        print(f'[{rendered}] saved {ours_path}')
        print(f'[{rendered}] saved {gt_path}')

        if args.max_samples > 0 and rendered >= args.max_samples:
            break

    if rendered == 0:
        raise RuntimeError('No samples were rendered. Check sample token or paths.')


if __name__ == '__main__':
    main()
