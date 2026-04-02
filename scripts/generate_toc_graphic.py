#!/usr/bin/env python3
"""Generate the ACS TOC graphic for the snake-scale biofilm manuscript."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / 'results' / 'submission' / 'figures'
PNG_PATH = OUT_DIR / 'figure_toc_graphic.png'
SVG_PATH = OUT_DIR / 'figure_toc_graphic.svg'

FIGSIZE = (3.25, 1.75)
DPI = 300
FONT_STACK = ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif']


def add_spike(ax, center, width, height, face='#e69a37', highlight='#fde5a4', alpha=1.0):
    cx, cy = center
    ax.add_patch(
        Polygon(
            [(cx - width / 2, cy), (cx, cy + height), (cx + width / 2, cy)],
            closed=True,
            facecolor=face,
            edgecolor='#fff5d1',
            linewidth=0.35,
            joinstyle='round',
            alpha=alpha,
            zorder=5,
        )
    )
    ax.add_patch(
        Polygon(
            [
                (cx - width * 0.10, cy + height * 0.18),
                (cx - width * 0.01, cy + height * 0.92),
                (cx + width * 0.16, cy + height * 0.18),
            ],
            closed=True,
            facecolor=highlight,
            edgecolor='none',
            alpha=0.82 * alpha,
            zorder=6,
        )
    )


def add_capsule(ax, center, length, width, angle, face, edge, highlight, alpha=1.0):
    cx, cy = center
    body = FancyBboxPatch(
        (cx - length / 2, cy - width / 2),
        length,
        width,
        boxstyle=f'round,pad=0.0,rounding_size={width / 2}',
        facecolor=face,
        edgecolor=edge,
        linewidth=0.55,
        alpha=alpha,
        zorder=8,
    )
    body.set_transform(transforms.Affine2D().rotate_deg_around(cx, cy, angle) + ax.transData)
    ax.add_patch(body)
    gloss = FancyBboxPatch(
        (cx - length * 0.22, cy - width * 0.03),
        length * 0.32,
        width * 0.13,
        boxstyle=f'round,pad=0.0,rounding_size={width / 2}',
        facecolor=highlight,
        edgecolor='none',
        alpha=0.42 * alpha,
        zorder=9,
    )
    gloss.set_transform(transforms.Affine2D().rotate_deg_around(cx, cy, angle) + ax.transData)
    ax.add_patch(gloss)


def add_eps_cloud(ax, x0, y0, x1, y1, rows, cols, broken=False, alpha=1.0):
    xs = [x0 + (x1 - x0) * i / max(cols - 1, 1) for i in range(cols)]
    ys = [y0 + (y1 - y0) * j / max(rows - 1, 1) for j in range(rows)]
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            if broken and (i + j) % 3 == 0:
                continue
            ax.add_patch(
                Circle(
                    (x, y),
                    radius=0.007 if not broken else 0.0055,
                    facecolor='#9b7bd6',
                    edgecolor='none',
                    alpha=(0.32 if not broken else 0.18) * alpha,
                    zorder=6,
                )
            )


def add_eps_fragments(ax, points, radius=0.005, alpha=0.22):
    for x, y in points:
        ax.add_patch(
            Circle(
                (x, y),
                radius=radius,
                facecolor='#9b7bd6',
                edgecolor='none',
                alpha=alpha,
                zorder=7,
            )
        )


def draw_subpanel_frame(ax, x, y, w, h):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle='round,pad=0.012,rounding_size=0.025',
            facecolor='#f5f2eb',
            edgecolor='#d6d1c8',
            linewidth=0.7,
            zorder=1,
        )
    )


def draw_flat_surface(ax, x, y, w, h, color):
    ax.add_patch(
        Rectangle(
            (x + w * 0.10, y + h * 0.16),
            w * 0.80,
            h * 0.10,
            facecolor=color,
            edgecolor='none',
            zorder=2,
        )
    )


def draw_smooth_subpanel(ax, x, y, w, h):
    draw_subpanel_frame(ax, x, y, w, h)
    draw_flat_surface(ax, x, y, w, h, '#c7c0b5')
    add_eps_cloud(ax, x + w * 0.18, y + h * 0.30, x + w * 0.82, y + h * 0.62, rows=4, cols=6)
    bacteria = [
        (x + w * 0.24, y + h * 0.49, 14),
        (x + w * 0.40, y + h * 0.57, -8),
        (x + w * 0.57, y + h * 0.50, 10),
        (x + w * 0.73, y + h * 0.58, -12),
        (x + w * 0.34, y + h * 0.37, 4),
        (x + w * 0.63, y + h * 0.36, -4),
    ]
    colors = [
        ('#4ba6a2', '#256d6b', '#d4f3ec'),
        ('#5d96d5', '#2f6198', '#dce9ff'),
        ('#63ba82', '#2e7f4e', '#dcf6de'),
        ('#4ba6a2', '#256d6b', '#d4f3ec'),
        ('#5d96d5', '#2f6198', '#dce9ff'),
        ('#63ba82', '#2e7f4e', '#dcf6de'),
    ]
    for (cx, cy, angle), color in zip(bacteria, colors):
        add_capsule(ax, (cx, cy), w * 0.22, h * 0.075, angle, *color)


def draw_spiked_subpanel(ax, x, y, w, h):
    draw_subpanel_frame(ax, x, y, w, h)
    draw_flat_surface(ax, x, y, w, h, '#d7d1c6')
    base_y = y + h * 0.26
    xs = [x + w * v for v in (0.20, 0.32, 0.44, 0.57, 0.69, 0.82)]
    heights = [0.12, 0.22, 0.15, 0.20, 0.16, 0.13]
    widths = [0.046, 0.050, 0.045, 0.048, 0.044, 0.043]
    offsets = [0.000, 0.004, -0.001, 0.003, -0.002, 0.001]
    for sx, rel_h, rel_w, dx in zip(xs, heights, widths, offsets):
        add_spike(ax, (sx + w * dx, base_y), w * rel_w, h * rel_h)
    add_eps_cloud(ax, x + w * 0.30, y + h * 0.36, x + w * 0.70, y + h * 0.53, rows=2, cols=4, broken=True, alpha=0.8)
    add_eps_fragments(
        ax,
        [
            (x + w * 0.36, y + h * 0.55),
            (x + w * 0.41, y + h * 0.50),
            (x + w * 0.53, y + h * 0.47),
            (x + w * 0.61, y + h * 0.51),
            (x + w * 0.67, y + h * 0.56),
        ],
        radius=0.0048,
        alpha=0.20,
    )
    bacteria = [
        (x + w * 0.30, y + h * 0.53, 8, 1.0),
        (x + w * 0.61, y + h * 0.56, -8, 1.0),
        (x + w * 0.47, y + h * 0.38, 2, 0.48),
        (x + w * 0.40, y + h * 0.44, -18, 0.26),
    ]
    colors = [
        ('#4ba6a2', '#256d6b', '#d4f3ec'),
        ('#5d96d5', '#2f6198', '#dce9ff'),
        ('#63ba82', '#2e7f4e', '#dcf6de'),
        ('#63ba82', '#2e7f4e', '#dcf6de'),
    ]
    for (cx, cy, angle, alpha), color in zip(bacteria, colors):
        add_capsule(ax, (cx, cy), w * 0.22, h * 0.075, angle, *color, alpha=alpha)


def add_labels(ax):
    ax.text(
        0.50,
        0.965,
        'topography limits early biofilm stabilisation',
        ha='center',
        va='top',
        fontsize=7.4,
        color='#28313c',
        fontfamily='sans-serif',
        zorder=20,
    )
    ax.text(
        0.245,
        0.028,
        'smooth substrate',
        ha='center',
        va='bottom',
        fontsize=6.2,
        color='#28313c',
        fontfamily='sans-serif',
        zorder=20,
    )
    ax.text(
        0.755,
        0.028,
        'microspike topography',
        ha='center',
        va='bottom',
        fontsize=6.2,
        color='#28313c',
        fontfamily='sans-serif',
        fontstyle='italic',
        zorder=20,
    )


def build_figure():
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = FONT_STACK

    fig = plt.figure(figsize=FIGSIZE, dpi=DPI, facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    draw_smooth_subpanel(ax, 0.05, 0.22, 0.39, 0.52)
    draw_spiked_subpanel(ax, 0.56, 0.22, 0.39, 0.52)
    ax.annotate(
        '',
        xy=(0.53, 0.48),
        xytext=(0.46, 0.48),
        arrowprops=dict(arrowstyle='-|>', lw=1.1, color='#6c7380'),
        zorder=15,
    )
    add_labels(ax)
    return fig


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    fig.savefig(PNG_PATH, dpi=DPI, facecolor='white')
    fig.savefig(SVG_PATH, dpi=DPI, facecolor='white')
    plt.close(fig)
    print(f'Saved {PNG_PATH}')
    print(f'Saved {SVG_PATH}')


if __name__ == '__main__':
    main()



