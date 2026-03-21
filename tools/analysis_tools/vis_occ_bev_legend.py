import argparse
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Tuple

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

from vis_occ_bev import REPO_ROOT, SEMANTIC_COLORS, SEMANTIC_NAMES


DEFAULT_OUTPUT_PATH = REPO_ROOT / 'bev_vis' / 'legend_horizontal.png'


@dataclass
class LegendItem:
    label: str
    color: Tuple[float, float, float]
    marker: str
    text_width: float = 0.0
    text_height: float = 0.0
    item_width: float = 0.0
    item_height: float = 0.0


def parse_args():
    parser = argparse.ArgumentParser(
        description='Render a standalone occupancy legend image with a '
        'wrapped horizontal layout.')
    parser.add_argument(
        '--output',
        default=str(DEFAULT_OUTPUT_PATH),
        help='Output image path.')
    parser.add_argument(
        '--dpi',
        type=int,
        default=600,
        help='DPI for the saved figure.')
    parser.add_argument(
        '--font-size',
        type=int,
        default=9,
        help='Legend label font size in points.')
    parser.add_argument(
        '--marker-size',
        type=int,
        default=15,
        help='Marker box diameter in pixels.')
    parser.add_argument(
        '--text-gap',
        type=int,
        default=4,
        help='Gap between marker and label in pixels.')
    parser.add_argument(
        '--item-gap',
        type=int,
        default=10,
        help='Gap between items in the same row in pixels.')
    parser.add_argument(
        '--row-gap',
        type=int,
        default=7,
        help='Gap between rows in pixels.')
    parser.add_argument(
        '--padding',
        type=int,
        default=14,
        help='Outer padding around the legend in pixels.')
    parser.add_argument(
        '--target-aspect',
        type=float,
        default=2.8,
        help='Preferred width/height ratio for the legend canvas.')
    parser.add_argument(
        '--min-rows',
        type=int,
        default=2,
        help='Minimum number of rows to consider.')
    parser.add_argument(
        '--max-rows',
        type=int,
        default=4,
        help='Maximum number of rows to consider.')
    parser.add_argument(
        '--exclude-lidar',
        action='store_true',
        help='Exclude the GT LiDAR entry.')
    return parser.parse_args()


def build_items(include_lidar):
    items = []
    if include_lidar:
        items.append(LegendItem(label='GT LiDAR', color=(0.0, 0.0, 0.0), marker='circle'))
    for name, color in zip(SEMANTIC_NAMES, SEMANTIC_COLORS):
        items.append(
            LegendItem(
                label=name.replace('_', ' '),
                color=tuple(color.tolist()),
                marker='square',
            ))
    return items


def measure_items(items, font_size, marker_size, text_gap, dpi):
    fig = plt.figure(figsize=(2, 2), dpi=dpi)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for item in items:
        text = fig.text(0, 0, item.label, fontsize=font_size, family='sans-serif')
        bbox = text.get_window_extent(renderer=renderer)
        text.remove()
        item.text_width = math.ceil(bbox.width)
        item.text_height = math.ceil(bbox.height)
        item.item_width = marker_size + text_gap + item.text_width
        item.item_height = max(marker_size, item.text_height)
    plt.close(fig)


def build_rows(items, breakpoints):
    return [items[start:end] for start, end in zip(breakpoints[:-1], breakpoints[1:])]


def partition_items(items, num_rows, item_gap):
    num_items = len(items)
    widths = [item.item_width for item in items]
    prefix_widths = [0.0]
    for width in widths:
        prefix_widths.append(prefix_widths[-1] + width)

    total_row_gaps = max(0, num_items - num_rows) * item_gap
    target_row_width = (prefix_widths[-1] + total_row_gaps) / num_rows

    def row_width(start_idx, end_idx):
        count = end_idx - start_idx
        width = prefix_widths[end_idx] - prefix_widths[start_idx]
        if count > 1:
            width += item_gap * (count - 1)
        return width

    @lru_cache(maxsize=None)
    def solve(start_idx, rows_left):
        items_left = num_items - start_idx
        if rows_left == 1:
            width = row_width(start_idx, num_items)
            return (width - target_row_width) ** 2, (num_items,)

        best = None
        max_end = num_items - (rows_left - 1)
        for end_idx in range(start_idx + 1, max_end + 1):
            width = row_width(start_idx, end_idx)
            row_cost = (width - target_row_width) ** 2
            tail_cost, tail_breaks = solve(end_idx, rows_left - 1)
            total_cost = row_cost + tail_cost
            if best is None or total_cost < best[0]:
                best = (total_cost, (end_idx,) + tail_breaks)
        return best

    _, break_tail = solve(0, num_rows)
    breakpoints = (0,) + break_tail
    return build_rows(items, breakpoints)


def choose_layout(items, row_gap, item_gap, padding, target_aspect,
                  min_rows, max_rows):
    best = None
    max_rows = min(max_rows, len(items))
    min_rows = max(1, min(min_rows, max_rows))
    for num_rows in range(min_rows, max_rows + 1):
        rows = partition_items(items, num_rows, item_gap)
        row_widths = []
        row_heights = []
        for row in rows:
            width = sum(item.item_width for item in row)
            if len(row) > 1:
                width += item_gap * (len(row) - 1)
            row_widths.append(width)
            row_heights.append(max(item.item_height for item in row))
        total_width = max(row_widths) + padding * 2
        total_height = sum(row_heights) + row_gap * (len(rows) - 1) + padding * 2
        aspect = total_width / total_height
        aspect_cost = abs(math.log(aspect / target_aspect))
        ragged_cost = (max(row_widths) - min(row_widths)) / max(row_widths)
        row_penalty = abs(num_rows - 3) * 0.03
        score = aspect_cost + ragged_cost * 0.35 + row_penalty
        if best is None or score < best['score']:
            best = {
                'score': score,
                'rows': rows,
                'row_widths': row_widths,
                'row_heights': row_heights,
                'total_width': total_width,
                'total_height': total_height,
            }
    return best


def render_legend(items, layout, out_path: Path, dpi, font_size, marker_size,
                  text_gap, item_gap, row_gap, padding):
    total_width = layout['total_width']
    total_height = layout['total_height']
    fig = plt.figure(
        figsize=(total_width / dpi, total_height / dpi),
        dpi=dpi,
        facecolor='white',
    )
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, total_width)
    ax.set_ylim(total_height, 0)
    ax.axis('off')

    current_y = padding
    for row, row_width, row_height in zip(
            layout['rows'], layout['row_widths'], layout['row_heights']):
        x = (total_width - row_width) / 2
        center_y = current_y + row_height / 2
        for item in row:
            if item.marker == 'circle':
                patch = Circle(
                    (x + marker_size / 2, center_y),
                    radius=marker_size / 2,
                    facecolor=item.color,
                    edgecolor='white',
                    linewidth=1.0,
                )
            else:
                patch = Rectangle(
                    (x, center_y - marker_size / 2),
                    marker_size,
                    marker_size,
                    facecolor=item.color,
                    edgecolor='none',
                )
            ax.add_patch(patch)
            ax.text(
                x + marker_size + text_gap,
                center_y,
                item.label,
                fontsize=font_size,
                va='center',
                ha='left',
                color=(0.1, 0.1, 0.1),
                family='sans-serif',
            )
            x += item.item_width + (0 if item is row[-1] else item_gap)
        current_y += row_height + row_gap

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor='white', bbox_inches=None, pad_inches=0)
    plt.close(fig)


def main():
    args = parse_args()
    out_path = Path(args.output)

    items = build_items(include_lidar=not args.exclude_lidar)
    measure_items(
        items,
        font_size=args.font_size,
        marker_size=args.marker_size,
        text_gap=args.text_gap,
        dpi=args.dpi,
    )
    layout = choose_layout(
        items,
        row_gap=args.row_gap,
        item_gap=args.item_gap,
        padding=args.padding,
        target_aspect=args.target_aspect,
        min_rows=args.min_rows,
        max_rows=args.max_rows,
    )
    render_legend(
        items,
        layout,
        out_path=out_path,
        dpi=args.dpi,
        font_size=args.font_size,
        marker_size=args.marker_size,
        text_gap=args.text_gap,
        item_gap=args.item_gap,
        row_gap=args.row_gap,
        padding=args.padding,
    )
    print(f'[saved] {out_path}')


if __name__ == '__main__':
    main()
