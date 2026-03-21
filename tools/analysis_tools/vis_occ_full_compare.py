import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import torch
from PIL import Image

from vis_occ import (
    FREE_LABEL,
    SEMANTIC_NAMES,
    VOXEL_SIZE,
    colormap_to_colors,
    draw_legend_multirow,
    show_occ,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BEV_VIS_DIR = REPO_ROOT / 'bev_vis'
DEFAULT_BASELINE_RESULTS_DIR = REPO_ROOT / 'work_dirs/reference_results'
DEFAULT_OURS_RESULTS_DIR = REPO_ROOT / 'work_dirs/primary_results'
DEFAULT_OUTPUT_DIR = REPO_ROOT / 'full_compare'
INFO_CANDIDATES = [
    REPO_ROOT / 'data/nuscenes/bevdetv2-nuscenes_infos_val.pkl',
    REPO_ROOT / 'data/nuscenes/bevdetv2-nuscenes_infos_train.pkl',
]
FONT_CANDIDATES = [
    Path('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'),
    Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'),
]
CAMERA_VIEWS = [
    'CAM_FRONT_LEFT',
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_BACK_LEFT',
    'CAM_BACK',
    'CAM_BACK_RIGHT',
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Render overall occupancy comparison images for samples '
        'listed under bev_vis.')
    parser.add_argument(
        '--bev-vis-dir',
        default=str(DEFAULT_BEV_VIS_DIR),
        help='Directory used to discover scene names and sample tokens.')
    parser.add_argument(
        '--baseline-results-dir',
        default=str(DEFAULT_BASELINE_RESULTS_DIR),
        help='Results directory for the baseline model.')
    parser.add_argument(
        '--ours-results-dir',
        default=str(DEFAULT_OURS_RESULTS_DIR),
        help='Results directory for the ours model.')
    parser.add_argument(
        '--output-dir',
        default=str(DEFAULT_OUTPUT_DIR),
        help='Directory to save overall comparison images.')
    parser.add_argument(
        '--sample-token',
        default=None,
        help='Only render one sample token when set.')
    parser.add_argument(
        '--scene-name',
        default=None,
        help='Only render one scene when set.')
    parser.add_argument(
        '--max-samples',
        type=int,
        default=0,
        help='Maximum number of matched samples to render. Use 0 for all.')
    parser.add_argument(
        '--canva-size',
        type=int,
        default=2000,
        help='Size of the occupancy panel in pixels.')
    parser.add_argument(
        '--scale-factor',
        type=int,
        default=2,
        help='Trade-off between camera views and occupancy panel size.')
    parser.add_argument(
        '--dpi',
        type=int,
        default=600,
        help='PNG DPI metadata written to overall outputs.')
    return parser.parse_args()


def resolve_path(path_str):
    path = Path(path_str)
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    repo_candidate = (REPO_ROOT / path).resolve()
    if repo_candidate.exists():
        return repo_candidate
    return repo_candidate


def resolve_font_path():
    for font_path in FONT_CANDIDATES:
        if font_path.exists():
            return str(font_path)
    raise FileNotFoundError(
        'Cannot find a usable font. Checked: '
        + ', '.join(str(path) for path in FONT_CANDIDATES))


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
            'Expected one of: ' + ', '.join(str(path) for path in INFO_CANDIDATES))
    return info_index


def scan_target_samples(bev_vis_dir: Path):
    sample_refs = set()
    for image_path in bev_vis_dir.rglob('*.png'):
        scene_name = image_path.parent.name
        if not scene_name.startswith('scene-'):
            continue
        stem = image_path.stem
        if '_' not in stem:
            continue
        sample_token = stem.rsplit('_', 1)[0]
        if sample_token:
            sample_refs.add((scene_name, sample_token))
    return sorted(sample_refs)


def load_prediction(pred_path: Path):
    data = np.load(pred_path, allow_pickle=True)
    pred = data['pred']
    sample_token = str(data['sample_token'].tolist())
    return pred, sample_token


def load_camera_mask(info):
    occ_path = resolve_path(Path(info['occ_path']) / 'labels.npz')
    gt_data = np.load(occ_path)
    if 'mask_camera' in gt_data.files:
        return gt_data['mask_camera']
    return np.ones(gt_data['semantics'].shape, dtype=bool)


def load_camera_images(info):
    images = []
    for view in CAMERA_VIEWS:
        image_path = resolve_path(info['cams'][view]['data_path'])
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f'Failed to read camera image: {image_path}')
        images.append(image)
    return images


def build_visualizer():
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(width=4000, height=4000, visible=True)
    return vis


def configure_visualizer(vis):
    view_control = vis.get_view_control()
    view_control.set_lookat(np.array([-0.185, 0.513, 3.485]))
    view_control.set_front(np.array([-0.974, -0.055, 0.221]))
    view_control.set_up(np.array([0.221, 0.014, 0.975]))
    view_control.set_zoom(0.08)

    opt = vis.get_render_option()
    opt.background_color = np.asarray([1.0, 1.0, 1.0])
    opt.line_width = 5


def render_occ_canvas(vis, pred_occ, camera_mask):
    voxel_show = np.logical_and(pred_occ != FREE_LABEL, camera_mask)
    show_occ(
        torch.from_numpy(pred_occ),
        torch.from_numpy(voxel_show),
        voxel_size=VOXEL_SIZE,
        vis=vis,
        offset=[0, 0, 0],
    )
    configure_visualizer(vis)
    for _ in range(3):
        vis.poll_events()
        vis.update_renderer()
    occ_canvas = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    vis.clear_geometries()
    occ_canvas = (occ_canvas * 255).astype(np.uint8)
    return occ_canvas[..., [2, 1, 0]]


def trim_occ_canvas(occ_canvas, background_threshold=245):
    content_mask = np.any(occ_canvas < background_threshold, axis=2)
    if not np.any(content_mask):
        return occ_canvas

    content_rows = np.any(content_mask, axis=1)
    content_cols = np.any(content_mask, axis=0)

    top = int(np.argmax(content_rows))
    bottom = int(len(content_rows) - np.argmax(content_rows[::-1]))
    left = int(np.argmax(content_cols))
    right = int(len(content_cols) - np.argmax(content_cols[::-1]))

    content_height = max(1, bottom - top)
    content_width = max(1, right - left)

    top_margin = max(8, content_height // 45)
    bottom_margin = max(2, content_height // 90)
    side_margin = max(8, content_width // 45)
    top = max(0, top - top_margin)
    bottom = min(occ_canvas.shape[0], bottom + bottom_margin)
    left = max(0, left - side_margin)
    right = min(occ_canvas.shape[1], right + side_margin)
    return occ_canvas[top:bottom, left:right, :]


def build_overall_image(images, occ_canvas, canva_size, scale_factor, font_path):
    front_view_height = int(900 / scale_factor)
    back_view_height = front_view_height
    total_width = int(1600 / scale_factor * 3)
    occ_panel_padding_x = max(24, total_width // 40)
    occ_size = max(1, min(canva_size, total_width - 2 * occ_panel_padding_x))
    occ_canvas = trim_occ_canvas(occ_canvas)
    occ_scale = min(
        occ_size / occ_canvas.shape[1],
        occ_size / occ_canvas.shape[0],
    )
    occ_render_width = max(1, int(round(occ_canvas.shape[1] * occ_scale)))
    occ_render_height = max(1, int(round(occ_canvas.shape[0] * occ_scale)))
    occ_canvas_resize = cv2.resize(
        occ_canvas,
        (occ_render_width, occ_render_height),
        interpolation=cv2.INTER_LANCZOS4)

    font_size = max(32, min(48, total_width // 50))
    legend_canvas = draw_legend_multirow(
        SEMANTIC_NAMES,
        colormap_to_colors[:, :3].astype(np.uint8),
        total_width,
        font_path=font_path,
        min_rows=3,
        max_rows=3,
        font_size=font_size,
        box_size=int(round(font_size * 1.1)),
        text_gap=max(14, font_size // 3),
        item_gap=max(24, total_width // 60),
        padding_x=max(20, total_width // 40),
        padding_y=max(12, font_size // 3),
        row_gap=max(10, font_size // 3),
    )

    occ_panel_top_padding = max(8, front_view_height // 30)
    occ_panel_bottom_padding = max(2, front_view_height // 120)
    occ_panel = np.ones(
        (
            occ_render_height + occ_panel_top_padding + occ_panel_bottom_padding,
            total_width,
            3,
        ),
        dtype=np.uint8,
    ) * 255
    occ_x_begin = (total_width - occ_render_width) // 2
    occ_panel[
        occ_panel_top_padding:occ_panel_top_padding + occ_render_height,
        occ_x_begin:occ_x_begin + occ_render_width,
        :,
    ] = occ_canvas_resize

    total_height = (
        front_view_height + legend_canvas.shape[0] +
        occ_panel.shape[0] + back_view_height)
    big_img = np.ones((total_height, total_width, 3), dtype=np.uint8) * 255

    front_img = cv2.resize(
        np.concatenate(images[:3], axis=1), (total_width, front_view_height))
    back_img = cv2.resize(
        np.concatenate(
            [images[3][:, ::-1, :], images[4][:, ::-1, :], images[5][:, ::-1, :]],
            axis=1),
        (total_width, back_view_height),
    )

    current_y = 0
    big_img[current_y:current_y + front_view_height, :, :] = front_img
    current_y += front_view_height
    big_img[current_y:current_y + legend_canvas.shape[0], :, :] = legend_canvas
    current_y += legend_canvas.shape[0]
    big_img[current_y:current_y + occ_panel.shape[0], :, :] = occ_panel
    current_y += occ_panel.shape[0]
    big_img[current_y:current_y + back_view_height, :, :] = back_img
    return big_img


def save_overall_image(image_bgr, out_path: Path, dpi: int):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    Image.fromarray(image_rgb).save(out_path, dpi=(dpi, dpi))


def render_and_save(vis, info, pred_path: Path, out_path: Path, args, font_path):
    pred_occ, pred_token = load_prediction(pred_path)
    if pred_token != info['token']:
        print(
            f'[warn] token mismatch in {pred_path}: '
            f'{pred_token} != {info["token"]}')
    camera_mask = load_camera_mask(info)
    images = load_camera_images(info)
    occ_canvas = render_occ_canvas(vis, pred_occ, camera_mask)
    overall_image = build_overall_image(
        images,
        occ_canvas,
        canva_size=args.canva_size,
        scale_factor=args.scale_factor,
        font_path=font_path,
    )
    save_overall_image(overall_image, out_path, dpi=args.dpi)


def main():
    args = parse_args()
    bev_vis_dir = Path(args.bev_vis_dir)
    baseline_results_dir = Path(args.baseline_results_dir)
    ours_results_dir = Path(args.ours_results_dir)
    output_dir = Path(args.output_dir)

    if not bev_vis_dir.exists():
        raise FileNotFoundError(f'bev_vis directory does not exist: {bev_vis_dir}')
    if not baseline_results_dir.exists():
        raise FileNotFoundError(
            f'Baseline results directory does not exist: {baseline_results_dir}')
    if not ours_results_dir.exists():
        raise FileNotFoundError(
            f'Ours results directory does not exist: {ours_results_dir}')

    sample_refs = scan_target_samples(bev_vis_dir)
    if not sample_refs:
        raise RuntimeError(f'No sample references found under {bev_vis_dir}')

    info_index = load_info_index()
    font_path = resolve_font_path()
    vis = build_visualizer()

    rendered_samples = 0
    saved_outputs = 0
    try:
        for scene_name, sample_token in sample_refs:
            if args.scene_name and scene_name != args.scene_name:
                continue
            if args.sample_token and sample_token != args.sample_token:
                continue

            info = info_index.get(sample_token)
            if info is None:
                print(f'[skip] missing dataset info for token {sample_token}')
                continue
            if info.get('scene_name') != scene_name:
                print(
                    f'[skip] scene mismatch for token {sample_token}: '
                    f'bev_vis={scene_name}, info={info.get("scene_name")}')
                continue

            baseline_pred = baseline_results_dir / scene_name / sample_token / 'pred.npz'
            ours_pred = ours_results_dir / scene_name / sample_token / 'pred.npz'

            saved_any = False

            if baseline_pred.exists():
                baseline_out = output_dir / scene_name / f'{sample_token}_baseline.png'
                render_and_save(vis, info, baseline_pred, baseline_out, args, font_path)
                print(f'[saved] {baseline_out}')
                saved_outputs += 1
                saved_any = True
            else:
                print(f'[skip] missing baseline result: {baseline_pred}')

            if ours_pred.exists():
                ours_out = output_dir / scene_name / f'{sample_token}_ours.png'
                render_and_save(vis, info, ours_pred, ours_out, args, font_path)
                print(f'[saved] {ours_out}')
                saved_outputs += 1
                saved_any = True
            else:
                print(f'[skip] missing ours result: {ours_pred}')

            if saved_any:
                rendered_samples += 1
            if args.max_samples > 0 and rendered_samples >= args.max_samples:
                break
    finally:
        vis.destroy_window()

    if saved_outputs == 0:
        raise RuntimeError('No samples were rendered. Check filters and paths.')


if __name__ == '__main__':
    main()
