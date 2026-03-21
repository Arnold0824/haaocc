import os

import mmcv
import open3d as o3d

import numpy as np
import torch
import pickle
import math
from typing import Tuple, List, Dict, Iterable
import argparse
import cv2
from PIL import Image, ImageDraw, ImageFont

NOT_OBSERVED = -1
FREE = 0
OCCUPIED = 1
FREE_LABEL = 17
BINARY_OBSERVED = 1
BINARY_NOT_OBSERVED = 0

VOXEL_SIZE = [0.4, 0.4, 0.4]
POINT_CLOUD_RANGE = [-40, -40, -1, 40, 40, 5.4]
SPTIAL_SHAPE = [200, 200, 16]
TGT_VOXEL_SIZE = [0.4, 0.4, 0.4]
TGT_POINT_CLOUD_RANGE = [-40, -40, -1, 40, 40, 5.4]
SEMANTIC_NAMES = [
    'void', 'barrier', 'bicycle', 'bus', 'car', 'construction vehicle',
    'motorcycle', 'pedestrian', 'traffic cone', 'trailer', 'truck',
    'drivable surface', 'other flat', 'sidewalk', 'terrain',
    'manmade', 'vegetation'
]
# SEMANTIC_NAMES = [
#     '未定义', '护栏', '自行车', '公交车', '小汽车', '工程车',
#     '摩托车', '行人', '锥桶', '挂车', '卡车',
#     '可行驶区域', '其他地面', '人行道', '地形',
#     '建筑结构', '植被'
# ]


colormap_to_colors = np.array(
    [
        [0,   0,   0, 255],  # 0 undefined
        [112, 128, 144, 255],  # 1 barrier  orange
        [220, 20, 60, 255],    # 2 bicycle  Blue
        [255, 127, 80, 255],   # 3 bus  Darkslategrey
        [255, 158, 0, 255],  # 4 car  Crimson
        [233, 150, 70, 255],   # 5 cons. Veh  Orangered
        [255, 61, 99, 255],  # 6 motorcycle  Darkorange
        [0, 0, 230, 255], # 7 pedestrian  Darksalmon
        [47, 79, 79, 255],  # 8 traffic cone  Red
        [255, 140, 0, 255],# 9 trailer  Slategrey
        [255, 99, 71, 255],# 10 truck Burlywood
        [0, 207, 191, 255],    # 11 drive sur  Green
        [175, 0, 75, 255],  # 12 other lat  nuTonomy green
        [75, 0, 75, 255],  # 13 sidewalk
        [112, 180, 60, 255],    # 14 terrain
        [222, 184, 135, 255],    # 15 manmade
        [0, 175, 0, 255],   # 16 vegeyation
], dtype=np.float32)


def split_items_evenly(items, num_rows):
    rows = []
    base = len(items) // num_rows
    remainder = len(items) % num_rows
    start = 0

    for row_idx in range(num_rows):
        row_size = base + (1 if row_idx < remainder else 0)
        rows.append(items[start:start + row_size])
        start += row_size

    return rows


def draw_legend_multirow(class_names,
                         colormap,
                         canvas_width,
                         font_path,
                         min_rows=2,
                         max_rows=3,
                         font_size=30,
                         box_size=30,
                         text_gap=10,
                         item_gap=30,
                         padding_x=40,
                         padding_y=20,
                         row_gap=16):
    font = ImageFont.truetype(font_path, font_size)
    legend_items = []

    for idx, name in enumerate(class_names):
        color = tuple(int(c) for c in np.array(colormap[idx][:3]).flatten().tolist())
        bbox = font.getbbox(name)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        item_width = box_size + text_gap + text_width
        item_height = max(box_size, text_height)
        legend_items.append((name, color, item_width, item_height, text_height))

    rows = None
    for row_count in range(min_rows, max_rows + 1):
        candidate_rows = split_items_evenly(legend_items, row_count)
        max_row_width = 0
        for row in candidate_rows:
            if not row:
                continue
            row_width = sum(item[2] for item in row) + item_gap * (len(row) - 1)
            max_row_width = max(max_row_width, row_width)
        if max_row_width <= canvas_width - 2 * padding_x:
            rows = candidate_rows
            break

    if rows is None:
        rows = split_items_evenly(legend_items, max_rows)

    row_height = max(item[3] for item in legend_items)
    canvas_height = padding_y * 2 + len(rows) * row_height + (len(rows) - 1) * row_gap
    image_pil = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))
    draw = ImageDraw.Draw(image_pil)

    y = padding_y
    for row in rows:
        row_width = sum(item[2] for item in row) + item_gap * (len(row) - 1)
        x = max(padding_x, (canvas_width - row_width) // 2)

        for name, color, item_width, _, text_height in row:
            box_y = y + (row_height - box_size) // 2
            text_y = y + (row_height - text_height) // 2 - 1

            draw.rectangle([x, box_y, x + box_size, box_y + box_size], fill=color)
            draw.text((x + box_size + text_gap, text_y), name, fill=(20, 20, 20), font=font)
            x += item_width + item_gap

        y += row_height + row_gap

    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
def debug_color_render():
    from PIL import Image, ImageDraw
    test_img = Image.new('RGB', (600, 60), (255, 255, 255))
    draw = ImageDraw.Draw(test_img)
    x = 10
    for idx, name in enumerate(SEMANTIC_NAMES):
        color = tuple(int(c) for c in colormap_to_colors[idx][:3])
        draw.rectangle([x, 10, x + 20, 30], fill=color)
        x += 30
    test_img.show()
def voxel2points(voxel, occ_show, voxelSize):
    """
    Args:
        voxel: (Dx, Dy, Dz)
        occ_show: (Dx, Dy, Dz)
        voxelSize: (dx, dy, dz)

    Returns:
        points: (N, 3) 3: (x, y, z)
        voxel: (N, ) cls_id
        occIdx: (x_idx, y_idx, z_idx)
    """
    occIdx = torch.where(occ_show)
    points = torch.cat((occIdx[0][:, None] * voxelSize[0] + POINT_CLOUD_RANGE[0], \
                        occIdx[1][:, None] * voxelSize[1] + POINT_CLOUD_RANGE[1], \
                        occIdx[2][:, None] * voxelSize[2] + POINT_CLOUD_RANGE[2]),
                       dim=1)      # (N, 3) 3: (x, y, z)
    return points, voxel[occIdx], occIdx


def voxel_profile(voxel, voxel_size):
    """
    Args:
        voxel: (N, 3)  3:(x, y, z)
        voxel_size: (vx, vy, vz)

    Returns:
        box: (N, 7) (x, y, z - dz/2, vx, vy, vz, 0)
    """
    centers = torch.cat((voxel[:, :2], voxel[:, 2][:, None] - voxel_size[2] / 2), dim=1)     # (x, y, z - dz/2)
    # centers = voxel
    wlh = torch.cat((torch.tensor(voxel_size[0]).repeat(centers.shape[0])[:, None],
                     torch.tensor(voxel_size[1]).repeat(centers.shape[0])[:, None],
                     torch.tensor(voxel_size[2]).repeat(centers.shape[0])[:, None]), dim=1)
    yaw = torch.full_like(centers[:, 0:1], 0)
    return torch.cat((centers, wlh, yaw), dim=1)


def rotz(t):
    """Rotation about the z-axis."""
    c = torch.cos(t)
    s = torch.sin(t)
    return torch.tensor([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])


def my_compute_box_3d(center, size, heading_angle):
    """
    Args:
        center: (N, 3)  3: (x, y, z - dz/2)
        size: (N, 3)    3: (vx, vy, vz)
        heading_angle: (N, 1)
    Returns:
        corners_3d: (N, 8, 3)
    """
    h, w, l = size[:, 2], size[:, 0], size[:, 1]
    center[:, 2] = center[:, 2] + h / 2
    l, w, h = (l / 2).unsqueeze(1), (w / 2).unsqueeze(1), (h / 2).unsqueeze(1)
    x_corners = torch.cat([-l, l, l, -l, -l, l, l, -l], dim=1)[..., None]
    y_corners = torch.cat([w, w, -w, -w, w, w, -w, -w], dim=1)[..., None]
    z_corners = torch.cat([h, h, h, h, -h, -h, -h, -h], dim=1)[..., None]
    corners_3d = torch.cat([x_corners, y_corners, z_corners], dim=2)
    corners_3d[..., 0] += center[:, 0:1]
    corners_3d[..., 1] += center[:, 1:2]
    corners_3d[..., 2] += center[:, 2:3]
    return corners_3d


def show_point_cloud(points: np.ndarray, colors=True, points_colors=None, bbox3d=None, voxelize=False,
                     bbox_corners=None, linesets=None, vis=None, offset=[0,0,0], large_voxel=True, voxel_size=0.4):
    """
    :param points: (N, 3)  3:(x, y, z)
    :param colors: false 不显示点云颜色
    :param points_colors: (N, 4）
    :param bbox3d: voxel grid (N, 7) 7: (center, wlh, yaw=0)
    :param voxelize: false 不显示voxel边界
    :param bbox_corners: (N, 8, 3)  voxel grid 角点坐标, 用于绘制voxel grid 边界.
    :param linesets: 用于绘制voxel grid 边界.
    :return:
    """
    if vis is None:
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window()
    if isinstance(offset, list) or isinstance(offset, tuple):
        offset = np.array(offset)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points+offset)
    if colors:
        pcd.colors = o3d.utility.Vector3dVector(points_colors[:, :3])
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1, origin=[0, 0, 0])

    voxelGrid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
    if large_voxel:
        vis.add_geometry(voxelGrid)
    else:
        vis.add_geometry(pcd)

    if voxelize:
        line_sets = o3d.geometry.LineSet()
        line_sets.points = o3d.open3d.utility.Vector3dVector(bbox_corners.reshape((-1, 3))+offset)
        line_sets.lines = o3d.open3d.utility.Vector2iVector(linesets.reshape((-1, 2)))
        line_sets.paint_uniform_color((0, 0, 0))
        vis.add_geometry(line_sets)

    vis.add_geometry(mesh_frame)

    # ego_pcd = o3d.geometry.PointCloud()
    # ego_points = generate_the_ego_car()
    # ego_pcd.points = o3d.utility.Vector3dVector(ego_points)
    # vis-r50-miou32.add_geometry(ego_pcd)

    return vis


def show_occ(occ_state, occ_show, voxel_size, vis=None, offset=[0, 0, 0]):
    """
    Args:
        occ_state: (Dx, Dy, Dz), cls_id
        occ_show: (Dx, Dy, Dz), bool
        voxel_size: [0.4, 0.4, 0.4]
        vis: Visualizer
        offset:

    Returns:

    """
    colors = colormap_to_colors / 255
    pcd, labels, occIdx = voxel2points(occ_state, occ_show, voxel_size)
    # pcd: (N, 3)  3: (x, y, z)
    # labels: (N, )  cls_id
    _labels = labels % len(colors)
    pcds_colors = colors[_labels]   # (N, 4)

    bboxes = voxel_profile(pcd, voxel_size)    # (N, 7)   7: (x, y, z - dz/2, dx, dy, dz, 0)
    bboxes_corners = my_compute_box_3d(bboxes[:, 0:3], bboxes[:, 3:6], bboxes[:, 6:7])      # (N, 8, 3)

    bases_ = torch.arange(0, bboxes_corners.shape[0] * 8, 8)
    edges = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]])  # lines along y-axis
    edges = edges.reshape((1, 12, 2)).repeat(bboxes_corners.shape[0], 1, 1)     # (N, 12, 2)
    # (N, 12, 2) + (N, 1, 1) --> (N, 12, 2)   此时edges中记录的是bboxes_corners的整体id: (0, N*8).
    edges = edges + bases_[:, None, None]

    vis = show_point_cloud(
        points=pcd.numpy(),
        colors=True,
        points_colors=pcds_colors,
        voxelize=True,
        bbox3d=bboxes.numpy(),
        bbox_corners=bboxes_corners.numpy(),
        linesets=edges.numpy(),
        vis=vis,
        offset=offset,
        large_voxel=True,
        voxel_size=0.4
    )
    return vis


def generate_the_ego_car():
    ego_range = [-2, -1, 0, 2, 1, 1.5]
    ego_voxel_size=[0.1, 0.1, 0.1]
    ego_xdim = int((ego_range[3] - ego_range[0]) / ego_voxel_size[0])
    ego_ydim = int((ego_range[4] - ego_range[1]) / ego_voxel_size[1])
    ego_zdim = int((ego_range[5] - ego_range[2]) / ego_voxel_size[2])
    temp_x = np.arange(ego_xdim)
    temp_y = np.arange(ego_ydim)
    temp_z = np.arange(ego_zdim)
    ego_xyz = np.stack(np.meshgrid(temp_y, temp_x, temp_z), axis=-1).reshape(-1, 3)
    ego_point_x = (ego_xyz[:, 0:1] + 0.5) / ego_xdim * (ego_range[3] - ego_range[0]) + ego_range[0]
    ego_point_y = (ego_xyz[:, 1:2] + 0.5) / ego_ydim * (ego_range[4] - ego_range[1]) + ego_range[1]
    ego_point_z = (ego_xyz[:, 2:3] + 0.5) / ego_zdim * (ego_range[5] - ego_range[2]) + ego_range[2]
    ego_point_xyz = np.concatenate((ego_point_y, ego_point_x, ego_point_z), axis=-1)
    ego_points_label =  (np.ones((ego_point_xyz.shape[0]))*16).astype(np.uint8)
    ego_dict = {}
    ego_dict['point'] = ego_point_xyz
    ego_dict['label'] = ego_points_label
    return ego_point_xyz


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize the predicted '
                                     'result of nuScenes')
    parser.add_argument(
        'res',
        nargs='?',
        default='work_dirs/results',
        help='Path to the predicted result directory')
    parser.add_argument(
        '--canva-size', type=int, default=2000, help='Size of canva in pixel')
    parser.add_argument(
        '--vis-frames',
        type=int,
        default=1000,
        help='Number of frames for visualization')
    parser.add_argument(
        '--scale-factor',
        type=int,
        default=2,
        help='Trade-off between image-view and bev in size of '
        'the visualized canvas')
    parser.add_argument(
        '--version',
        type=str,
        default='val',
        help='Version of nuScenes dataset')
    parser.add_argument('--draw-gt', action='store_true')
    parser.add_argument(
        '--root_path',
        type=str,
        default='./data/nuscenes',
        help='Path to nuScenes dataset')
    parser.add_argument(
        '--save_path',
        type=str,
        default='./vis_occ',
        help='Path to save visualization results')
    parser.add_argument(
        '--format',
        type=str,
        default='image',
        choices=['video', 'image'],
        help='The desired format of the visualization result')
    parser.add_argument(
        '--fps', type=int, default=10, help='Frame rate of video')
    parser.add_argument(
        '--video-prefix', type=str, default='vis_occ', help='name of video')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # load predicted results
    results_dir = args.res

    # load dataset information
    info_path = 'data/nuscenes/bevdetv2-nuscenes_infos_%s.pkl' % args.version
    dataset = pickle.load(open(info_path, 'rb'))
    # prepare save path and medium
    vis_dir = args.save_path
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    overall_summary_dir = None
    if args.format == 'image':
        overall_summary_dir = os.path.join(vis_dir, 'overall')
        mmcv.mkdir_or_exist(overall_summary_dir)
    print('saving visualized result to %s' % vis_dir)
    scale_factor = args.scale_factor
    canva_size = args.canva_size
    if args.format == 'video':
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        vout = cv2.VideoWriter(
            os.path.join(vis_dir, '%s.mp4' % args.video_prefix), fourcc,
            args.fps, (int(1600 / scale_factor * 3),
                       int(900 / scale_factor * 2 + canva_size)))

    views = [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ]
    print('start visualizing results')

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(width=4000, height=4000, visible=True)

    for cnt, info in enumerate(
            dataset['infos'][:min(args.vis_frames, len(dataset['infos']))]):
        if cnt % 10 == 0:
            print('%d/%d' % (cnt, min(args.vis_frames, len(dataset['infos']))))

        scene_name = info['scene_name']
        sample_token = info['token']

        pred_occ_path = os.path.join(results_dir, scene_name, sample_token, 'pred.npz')
        gt_occ_path = info['occ_path']

        pred_occ = np.load(pred_occ_path)['pred']
        gt_data = np.load(os.path.join(gt_occ_path, 'labels.npz'))
        voxel_label = gt_data['semantics']
        lidar_mask = gt_data['mask_lidar']
        camera_mask = gt_data['mask_camera']

        # load imgs
        imgs = []
        for view in views:
            img = cv2.imread(info['cams'][view]['data_path'])
            imgs.append(img)

        # occ_canvas
        voxel_show = np.logical_and(pred_occ != FREE_LABEL, camera_mask)
        # voxel_show = pred_occ != FREE_LABEL
        voxel_size = VOXEL_SIZE
        vis = show_occ(torch.from_numpy(pred_occ), torch.from_numpy(voxel_show), voxel_size=voxel_size, vis=vis,
                       offset=[0, pred_occ.shape[0] * voxel_size[0] * 1.2 * 0, 0])

        if args.draw_gt:
            voxel_show = np.logical_and(voxel_label != FREE_LABEL, camera_mask)
            vis = show_occ(torch.from_numpy(voxel_label), torch.from_numpy(voxel_show), voxel_size=voxel_size, vis=vis,
                           offset=[0, voxel_label.shape[0] * voxel_size[0] * 1.2 * 1, 0])

        view_control = vis.get_view_control()

        look_at = np.array([-0.185, 0.513, 3.485])
        front = np.array([-0.974, -0.055, 0.221])
        up = np.array([0.221, 0.014, 0.975])
        zoom = np.array([0.08])

        view_control.set_lookat(look_at)
        view_control.set_front(front)
        view_control.set_up(up)
        view_control.set_zoom(zoom)

        opt = vis.get_render_option()
        opt.background_color = np.asarray([1, 1, 1])
        opt.line_width = 5

        vis.poll_events()
        vis.update_renderer()
        vis.run()

        # if args.format == 'image':
        #     out_dir = os.path.join(vis_dir, f'{scene_name}', f'{sample_token}')
        #     mmcv.mkdir_or_exist(out_dir)
        #     vis-r50-miou32.capture_screen_image(os.path.join(out_dir, 'screen_occ.png'), do_render=True)

        occ_canvas = vis.capture_screen_float_buffer(do_render=True)
        occ_canvas = np.asarray(occ_canvas)
        occ_canvas = (occ_canvas * 255).astype(np.uint8)
        occ_canvas = occ_canvas[..., [2, 1, 0]]

        vis.clear_geometries()

        front_view_height = int(900 / scale_factor)
        back_view_height = front_view_height
        total_width = int(1600 / scale_factor * 3)
        occ_panel_padding_x = max(24, total_width // 40)
        occ_panel_padding_y = max(24, front_view_height // 12)
        occ_size = max(1, min(canva_size, total_width - 2 * occ_panel_padding_x))
        occ_canvas_resize = cv2.resize(occ_canvas, (occ_size, occ_size), interpolation=cv2.INTER_LANCZOS4)

        font_size = max(28, min(42, total_width // 56))
        box_size = font_size
        colormap_for_legend = colormap_to_colors[:, :3].astype(np.uint8)
        legend_canvas = draw_legend_multirow(
            SEMANTIC_NAMES,
            colormap_for_legend,
            total_width,
            font_path="/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            min_rows=3,
            max_rows=3,
            font_size=font_size,
            box_size=box_size,
            text_gap=max(12, font_size // 3),
            item_gap=max(20, total_width // 64),
            padding_x=max(20, total_width // 40),
            padding_y=max(18, font_size // 2),
            row_gap=max(12, font_size // 2),
        )

        occ_panel = np.ones((occ_size + occ_panel_padding_y * 2, total_width, 3), dtype=np.uint8) * 255
        occ_x_begin = (total_width - occ_size) // 2
        occ_panel[occ_panel_padding_y:occ_panel_padding_y + occ_size,
                  occ_x_begin:occ_x_begin + occ_size, :] = occ_canvas_resize

        total_height = front_view_height + legend_canvas.shape[0] + occ_panel.shape[0] + back_view_height
        big_img = np.ones((total_height, total_width, 3), dtype=np.uint8) * 255

        front_img = cv2.resize(np.concatenate(imgs[:3], axis=1), (total_width, front_view_height))
        back_img = cv2.resize(
            np.concatenate([imgs[3][:, ::-1, :], imgs[4][:, ::-1, :], imgs[5][:, ::-1, :]], axis=1),
            (total_width, back_view_height))

        current_y = 0
        big_img[current_y:current_y + front_view_height, :, :] = front_img
        current_y += front_view_height
        big_img[current_y:current_y + legend_canvas.shape[0], :, :] = legend_canvas
        current_y += legend_canvas.shape[0]
        big_img[current_y:current_y + occ_panel.shape[0], :, :] = occ_panel
        current_y += occ_panel.shape[0]
        big_img[current_y:current_y + back_view_height, :, :] = back_img

        if args.format == 'image':
            out_dir = os.path.join(vis_dir, f'{scene_name}', f'{sample_token}')
            mmcv.mkdir_or_exist(out_dir)
            for i, img in enumerate(imgs):
                cv2.imwrite(os.path.join(out_dir, f'img{i}.png'), img)
            cv2.imwrite(os.path.join(out_dir, 'occ.png'), occ_canvas)
            cv2.imwrite(os.path.join(out_dir, 'overall.png'), big_img)
            summary_name = f'{scene_name}-{sample_token}.png'
            cv2.imwrite(os.path.join(overall_summary_dir, summary_name), big_img)
        elif args.format == 'video':
            cv2.putText(big_img, f'{cnt:{cnt}}', (5, 15), fontFace=cv2.FONT_HERSHEY_COMPLEX, color=(0, 0, 0),
                        fontScale=0.5)
            cv2.putText(big_img, f'{scene_name}', (5, 35), fontFace=cv2.FONT_HERSHEY_COMPLEX, color=(0, 0, 0),
                        fontScale=0.5)
            cv2.putText(big_img, f'{sample_token[:5]}', (5, 55), fontFace=cv2.FONT_HERSHEY_COMPLEX, color=(0, 0, 0),
                        fontScale=0.5)
            vout.write(big_img)

    if args.format == 'video':
        vout.release()
    vis.destroy_window()


if __name__ == '__main__':
    main()
