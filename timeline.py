"""Draw the contract and electricity market timeline with editable vector shapes.

Positions in the layout are points measured from the upper left of the diagram.
Icon routines use a common 40 x 40 local coordinate system.
The published artwork is Figs/TimeLine_V2.pdf; Figs/source/timeline.svg preserves
its layout. This independent Matplotlib redraw writes Figs/timeline_python.pdf.
"""

from math import cos, hypot, pi, sin

import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, PathPatch, Polygon, Rectangle
from matplotlib.transforms import Affine2D


NAVY = "#002b63"
INK = "#141b22"
HEADING = "#284f86"
BLUE = "#4472c4"
ORANGE = "#ed7d31"
BROWN = "#954510"


class Icon:
    """Translate and scale a pictogram while keeping stroke weights proportional."""

    def __init__(self, ax, x, y, width, height):
        self.ax = ax
        self.transform = Affine2D().scale(width / 40, height / 40).translate(x, y) + ax.transData
        self.scale = (width + height) / 80

    def patch(self, shape):
        shape.set_transform(self.transform)
        shape.set_linewidth(shape.get_linewidth() * self.scale)
        self.ax.add_patch(shape)

    def line(self, points, color=NAVY, width=1):
        path = Path(points)
        self.patch(
            PathPatch(
                path,
                fill=False,
                edgecolor=color,
                linewidth=width,
                capstyle="round",
                joinstyle="round",
            )
        )

    def box(self, x, y, w, h, fill, edge=NAVY, width=0.7, radius=0):
        style = dict(facecolor=fill, edgecolor=edge, linewidth=width)
        if radius:
            self.patch(
                FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}", **style)
            )
        else:
            self.patch(Rectangle((x, y), w, h, **style))

    def polygon(self, points, fill, edge=NAVY, width=0.8):
        self.patch(
            Polygon(
                points,
                closed=True,
                facecolor=fill,
                edgecolor=edge,
                linewidth=width,
                joinstyle="round",
            )
        )

    def circle(self, x, y, r, fill="none", edge=NAVY, width=0.8):
        self.patch(Circle((x, y), r, facecolor=fill, edgecolor=edge, linewidth=width))

    def curve(self, start, segments, fill="none", edge=NAVY, width=1):
        vertices = [start] + [pt for segment in segments for pt in segment]
        codes = [Path.MOVETO] + [Path.CURVE4] * (len(vertices) - 1)
        self.patch(
            PathPatch(
                Path(vertices, codes),
                facecolor=fill,
                edgecolor=edge,
                linewidth=width,
                capstyle="round",
                joinstyle="round",
            )
        )


def operator(i):
    """Cloud, communication terminals and the VPPO server rack."""
    i.curve(
        (9, 22),
        [
            ((0, 23), (-1, 15), (4, 12)),
            ((2, 1), (14, -3), (21, 5)),
            ((29, -1), (38, 4), (38, 11)),
            ((43, 16), (39, 23), (34, 22)),
        ],
        width=0.85,
    )
    i.line([(9, 22), (12, 22)])
    i.line([(34, 22), (29, 22)])
    for x, y, route in [
        (15.8, 8.8, [(19.7, 14), (19.7, 25)]),
        (11.8, 16.1, [(16.2, 20.4), (16.2, 25)]),
        (27.4, 13.2, [(23.3, 17.2), (23.3, 25)]),
    ]:
        i.line([(x, y), *route], width=0.85)
        i.circle(x, y, 2.3, "#f4f7fb", width=0.85)
    for y in (25, 30.1, 35.2):
        i.box(7.8, y, 24.4, 4.8, "#f4f7fb", width=0.9, radius=1.8)
        for x in (10.7, 13.8, 16.9):
            i.line([(x, y + 2.5), (x + 1.1, y + 2.5)], width=0.9)
        i.line([(23.0, y + 2.5), (29.5, y + 2.5)], width=0.85)


def green_building(i):
    """Building, sapling and three circular efficiency arrows."""
    for start, end in ((-57, -116), (207, 149), (27, -28)):
        angles = [pi * (start + (end - start) * j / 14) / 180 for j in range(15)]
        points = [(20 + 18.8 * cos(a), 20 + 18.8 * sin(a)) for a in angles]
        i.line(points, "#111111", 3.4)
        i.line(points, "#fbdb80", 1.8)
        x, y = points[-1]
        a = angles[-1] - pi / 2
        direction = (cos(a), sin(a))
        normal = (-direction[1], direction[0])
        i.polygon(
            [
                (x + 3.0 * direction[0], y + 3.0 * direction[1]),
                (x - 1.8 * direction[0] + 2.8 * normal[0], y - 1.8 * direction[1] + 2.8 * normal[1]),
                (x - 1.8 * direction[0] - 2.8 * normal[0], y - 1.8 * direction[1] - 2.8 * normal[1]),
            ],
            "#fbdb80",
            "#111111",
            0.8,
        )
    i.box(10.0, 11, 13.9, 25.5, "#004c63", "#111111", 0.9)
    for y in (13.5, 19, 24.5, 30):
        for x in (12.4, 18.1):
            i.box(x, y, 3.1, 5.9 if (x, y) == (18.1, 30) else 3.1, "white", "#121212", 0.65)
    i.curve(
        (28.5, 31.5),
        [((22.8, 29), (22.2, 19.5), (28.3, 17.9)), ((35.1, 21.3), (33.5, 29.4), (28.5, 31.5))],
        "#baef78",
        "#161616",
        0.9,
    )
    i.line([(28.6, 36.4), (28.6, 22.3)], "#161616", 0.85)
    i.line([(26.1, 25.9), (28.6, 28.3), (31, 25.7)], "#161616", 0.75)
    i.box(1.0, 36.5, 37.6, 2.8, NAVY, "#101820", 0.9)


def solar(i):
    i.circle(20, 8.2, 5.1, "#ffcf70", "#252525", 1.05)
    for angle in range(180, 361, 30):
        a = pi * angle / 180
        i.line(
            [(20 + 7.4 * cos(a), 8.2 + 7.4 * sin(a)), (20 + 9.4 * cos(a), 8.2 + 9.4 * sin(a))],
            "#252525",
            0.9,
        )
    i.box(18.2, 29.3, 4.1, 6.2, "#80b6d8", "#172637", 0.9)
    i.box(14.2, 35, 12, 2.2, "#63839b", "#172637", 1, radius=0.65)
    i.box(11.5, 37.2, 17.2, 2.3, "#46627b", "#172637", 1, radius=0.55)
    i.polygon([(3, 12.5), (36.7, 12.5), (39.3, 29.5), (0.7, 29.5)], "#87c9f3", "#171c22", 1.15)
    for y in (17, 22, 26.5):
        inset = (29.5 - y) * 2.3 / 17
        i.line([(0.8 + inset, y), (39.2 - inset, y)], "#223445", 0.85)
    for top, bottom in ((13.6, 11.7), (24.6, 26.3)):
        i.line([(top, 12.8), (bottom, 29.2)], "#253d53", 1.0)
    i.line([(2.5, 28), (38.3, 28)], "#e7f5ff", 0.8)


def storage(i):
    steel, face = "#465c70", "#d8e7f4"
    i.line([(1, 0.8), (13.2, 0.8), (13.2, 13.6)], steel, 1.05)
    i.line([(38.7, 0.8), (27, 0.8), (27, 13.2)], steel, 1.05)
    i.polygon([(1, 17), (9.6, 8.9), (39, 8.9), (30.2, 17)], "#b8d0e7", steel, 1.1)
    i.polygon([(30.2, 17), (39, 8.9), (39, 30.1), (30.2, 39.2)], "#9ab8d3", steel, 1.05)
    i.box(1, 17, 29.2, 22.2, face, steel, 1.05)
    i.polygon([(8.1, 15.1), (11.2, 11.9), (30, 11.9), (26.9, 15.1)], "#87a8c0", steel, 0.6)
    for x in (1.3, 11, 20.8):
        i.box(x + 0.3, 18.3, 8.7, 19.9, "#c4d9ee", steel, 0.55)
        for y in (20.7, 22.8, 24.9):
            i.line([(x + 1.7, y), (x + 7.4, y - 0.2)], "#5280aa", 0.55)
        i.box(x + 2, 31, 4.9, 3.2, "#8eb8d8", steel, 0.65)
    i.patch(
        Ellipse(
            (35.1, 23.3), 4.8, 8.2, angle=12, facecolor="#74a3c9", edgecolor=steel, linewidth=0.75
        )
    )
    i.line([(35.2, 20.9), (34.6, 23.7), (36, 23.2), (35.4, 25.7)], NAVY, 0.55)


def plant(i):
    steel = "#3e5361"
    for x, y in ((8, 8), (12, 3.8), (25, 7.3), (28.2, 3.4)):
        i.curve(
            (x - 2, y + 5),
            [
                ((x - 4, y + 1), (x + 1, y + 1), (x - 0.1, y - 2.6)),
                ((x + 1, y - 3.4), (x + 4, y - 3), (x + 4.5, y - 4.4)),
                ((x + 6, y - 0.6), (x + 1, y + 1.5), (x + 2, y + 4.2)),
            ],
            "#e1e6e7",
            steel,
            0.65,
        )
    for x in (2, 20.8):
        i.polygon(
            [
                (x + 4, 13),
                (x + 12, 13),
                (x + 11.6, 17.2),
                (x + 15.5, 26.8),
                (x, 26.8),
                (x + 4.3, 17.2),
            ],
            "#e2e7e9",
            steel,
            0.9,
        )
        i.line([(x + 3.8, 13.6), (x + 12.1, 13.6)], steel, 1.15)
        i.line([(x + 3.6, 19.3), (x + 12, 19.3)], "#b8795e", 1.05)
    i.box(10.1, 18.6, 15.7, 20.6, "#eed587", steel, 0.85)
    i.box(12.5, 20.4, 11.5, 2.2, "#dd935e", steel, 0.6)
    for x, y, w in ((1.2, 28.3, 10.8), (23.2, 28.8, 15.3)):
        i.box(x, y, w, 10.2, "#94d3dc", steel, 0.85)
        for j in range(3):
            for k in range(2):
                i.box(x + 1.4 + j * (w - 2.3) / 3, y + 1.5 + k * 3, 1.6, 1.5, "#65abc8", "none")
    for y in (25.4, 28.4, 31.4, 34.4):
        i.box(14.5, y, 2.6, 1.4, "#7699ac", "none")
    i.box(17.5, 33.8, 3.4, 5.1, "#93b2bf", steel, 0.55)


def pylon(i):
    steel, shadow = "#53646a", "#90a4a8"
    for points in [
        [(20, 0.5), (14.4, 6.1), (15.2, 12.2), (11.9, 38.5)],
        [(20, 0.5), (25.6, 6.1), (24.8, 12.2), (28.1, 38.5)],
    ]:
        i.line(points, shadow, 1.85)
        i.line(points, steel, 0.82)
    for y, w in ((7.7, 13), (12.0, 13), (17.2, 13)):
        for sign in (-1, 1):
            i.line([(20 + sign * 4.5, y - 3.5), (20 + sign * w, y), (20 + sign * 5, y)], steel, 1.1)
            i.line([(20 + sign * 10.4, y), (20 + sign * 10.4, y + 1.8)], steel, 0.7)
            i.line([(20 + sign * 12.4, y + 1.8), (20 + sign * 9.2, y + 1.8)], "#77b1ac", 0.9)
    for y0, y1, w0, w1 in (
        (6, 12, 5, 4.8),
        (12, 17, 4.8, 5.1),
        (17, 26.4, 5.1, 6.1),
        (26.4, 38.1, 6.1, 8),
    ):
        i.line([(20 - w0, y0), (20 + w1, y1)], steel, 1.15)
        i.line([(20 + w0, y0), (20 - w1, y1)], shadow, 1.15)
    i.box(7, 38.5, 26, 1.35, "#514b49", "#302c2b", 0.4)


def electricity(i):
    for angle in (-56, 125):
        points = [
            (20 + 17 * cos(pi * (angle + v) / 180), 20 + 17 * sin(pi * (angle + v) / 180))
            for v in range(0, 150, 8)
        ]
        i.line(points, "#343734", 1.2)
        x, y = points[-1]
        a = (angle + 144) * pi / 180
        i.line(
            [(x + 5 * cos(a - 0.3), y + 5 * sin(a - 0.3)), (x, y), (x - 4 * sin(a), y + 4 * cos(a))],
            "#343734",
            1.2,
        )
    i.polygon(
        [(22, 3.7), (14.4, 20.7), (19.2, 20.7), (16.8, 34.6), (26.2, 17.2), (21.2, 17.2)],
        "#ffcf64",
        "#262724",
        1.25,
    )


def arrow(ax, start, end, color, head=5.5, width=0.9, both=False):
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = hypot(dx, dy)
    ux, uy = dx / length, dy / length
    tops = [(end, 1)]
    if both:
        tops.append((start, -1))
    for (x, y), direction in tops:
        cx, cy = x - direction * head * ux, y - direction * head * uy
        points = [
            (x, y),
            (cx - uy * head / 2, cy + ux * head / 2),
            (cx + uy * head / 2, cy - ux * head / 2),
        ]
        ax.add_patch(Polygon(points, closed=True, color=color, linewidth=0))
    left = head * 0.82 if both else 0
    ax.plot(
        [start[0] + left * ux, end[0] - 0.82 * head * ux],
        [start[1] + left * uy, end[1] - 0.82 * head * uy],
        color=color,
        linewidth=width,
        solid_capstyle="butt",
    )


def exchange(ax, left, right, y, height, fill, edge):
    tip = height * 0.47
    waist = height * 0.24
    half = height / 2
    points = [
        (left, y),
        (left + tip, y - half),
        (left + tip, y - waist),
        (right - tip, y - waist),
        (right - tip, y - half),
        (right, y),
        (right - tip, y + half),
        (right - tip, y + waist),
        (left + tip, y + waist),
        (left + tip, y + half),
    ]
    ax.add_patch(Polygon(points, closed=True, facecolor=fill, edgecolor=edge, linewidth=1))


def brace(ax, left, right, center, y, tip, color, radius=7.2):
    mid = (y + tip) / 2
    dz = mid - y
    verts = [
        (left, y),
        (left, y + 0.55 * dz),
        (left + 0.45 * radius, mid),
        (left + radius, mid),
        (center - radius, mid),
        (center - 0.45 * radius, mid),
        (center, tip - 0.55 * dz),
        (center, tip),
        (center, tip - 0.55 * dz),
        (center + 0.45 * radius, mid),
        (center + radius, mid),
        (right - radius, mid),
        (right - 0.45 * radius, mid),
        (right, y + 0.55 * dz),
        (right, y),
    ]
    codes = (
        [Path.MOVETO]
        + [Path.CURVE4] * 3
        + [Path.LINETO]
        + [Path.CURVE4] * 6
        + [Path.LINETO]
        + [Path.CURVE4] * 3
    )
    ax.add_patch(PathPatch(Path(verts, codes), fill=False, linewidth=1.5, edgecolor=color))


def label(ax, x, y, words, size=19.8, color=INK, italic=False, bold=True):
    ax.text(
        x,
        y,
        words,
        fontsize=size,
        color=color,
        fontweight="bold" if bold else "normal",
        fontstyle="italic" if italic else "normal",
        va="baseline",
        ha="left",
    )


def panel(ax, x, y, width, height, fill, border):
    ax.add_patch(
        Rectangle(
            (x, y),
            width,
            height,
            facecolor=fill,
            edgecolor=border,
            linewidth=1.5,
            linestyle=(0, (4, 3)),
        )
    )


def negotiation(ax, dx, dy):
    operator(Icon(ax, 28.3 + dx, 103.4 + dy, 44.2, 46.7))
    green_building(Icon(ax, 99.8 + dx, 74.9 + dy, 49.7, 47.5))
    storage(Icon(ax, 101.8 + dx, 142.6 + dy, 42.4, 42.4))
    for y in (99.2, 163.6):
        arrow(ax, (75 + dx, 125 + dy), (101 + dx, y + dy), BLUE, head=6, width=0.45, both=True)


def coalition(ax, lower=False):
    if lower:
        cx, cy, fill, outline, inside = 421.83, 390.66, "#f4b183", "#f4b183", "#fbe5d6"
        icons = [
            (solar, 385.6, 346.1, 40.2, 40.8),
            (storage, 428.2, 350.6, 42.4, 42.4),
            (green_building, 369.0, 390.4, 38.2, 36.6),
            (plant, 416.9, 399.5, 32.7, 32.7),
        ]
    else:
        cx, cy, fill, outline, inside = 395.44, 121.53, BLUE, "#172c51", "#f2f2f2"
        icons = [
            (solar, 359.2, 76.9, 40.2, 40.8),
            (storage, 407.1, 79.9, 42.4, 42.4),
            (green_building, 342.6, 121.3, 38.2, 36.6),
            (plant, 395.8, 124.4, 37.1, 37.1),
        ]
    ax.add_patch(Ellipse((cx, cy), 185.72, 121.30, facecolor=fill, edgecolor=outline, linewidth=1))
    ax.add_patch(Ellipse((cx, cy), 172.16, 107.74, facecolor=inside, edgecolor=outline, linewidth=1))
    for draw, x, y, w, h in icons:
        draw(Icon(ax, x, y, w, h))


def market_timeline():
    width, height = 704.41, 456
    fig = plt.figure(figsize=(width / 72, height / 72))
    ax = fig.add_axes((0, 0, 1, 1), xlim=(0, width), ylim=(height, 0), aspect="equal")
    ax.set_axis_off()
    panel(ax, 19.36, 31.7, 639.86, 156.8, "#f4f6fb", "#8faadc")
    panel(ax, 95.4, 323.7, 481.9, 130.2, "#fef7f2", "#fbe5d6")
    ax.add_patch(Rectangle((19.36, 1.9), 639.86, 29.1, color="#dae3f3", linewidth=0))
    ax.add_patch(Rectangle((94.4, 294.6), 483.3, 29, color="#fbe5d6", linewidth=0))
    for x in (158.5, 520.1):
        ax.plot([x, x], [31.9, 181], color="#4682b4", linewidth=1.5, linestyle=(0, (4, 3)))
    label(ax, 253.5, 23.8, "Contract Market", 23.5, HEADING)
    label(ax, 224.3, 53.6, "Fixed Coalition Structure", 21, HEADING)
    for dx, dy in ((0, 0), (501.3, -1.8)):
        label(ax, 49.6 + dx, 52.8 + dy, "Contract", 20.7, HEADING)
        label(ax, 38.1 + dx, 72.1 + dy, "Negotiation", 20.7, HEADING)
    negotiation(ax, 0, 0)
    negotiation(ax, 501.1, -2.6)
    operator(Icon(ax, 189.8, 88.3, 53.6, 56.6))
    exchange(ax, 237.3, 301.7, 127.8, 23.2, BLUE, "#172c51")
    coalition(ax)
    for x, y, name in [
        (23.6, 171.3, "VPPO"),
        (97.3, 137.5, "DERs"),
        (190.8, 178.4, "VPPO"),
        (267.7, 178.4, "DERs"),
        (526.7, 169.0, "VPPO"),
        (600.4, 135.9, "DERs"),
    ]:
        label(ax, x, y, name)
    brace(ax, 19.36, 158.49, 86.70, 188.36, 217.49, "black")
    brace(ax, 520.09, 659.22, 587.43, 188.36, 218.4, "black", 7.9)
    brace(ax, 94.80, 585.77, 332.41, 264.85, 235.73, BLUE)
    for x in (6.8, 507.2):
        label(ax, x, 227.4, "(Switching Window)", 19.5, HEADING)
    label(ax, 251.6, 231.5, "(Contract Period ", 20.2, HEADING)
    label(ax, 403.1, 231.5, "t", 20.2, HEADING, italic=True)
    label(ax, 408.7, 231.5, ")", 20.2, HEADING)
    for x, sign in ((35.0, " − 1"), (602.9, " + 1")):
        label(ax, x, 257, "t", 20.5, HEADING, italic=True)
        label(ax, x + 5.7, 257, sign, 19.8, HEADING)
    for x in (83.4, 93.7, 585.5, 594.5):
        arrow(ax, (x, 265.5), (x, 231.3), BLUE, 6, 1.5)
    for x in [96.4 + 28.246 * j for j in range(18)] + [601.6, 629.8, 658.0, 686.2]:
        arrow(ax, (x, 265.1), (x, 280.2), ORANGE, 6, 1.5)
    arrow(ax, (1.15, 265.8), (703.9, 265.8), "black", 9, 3)
    label(ax, 7.9, 285.0, "Timeline", 20.7)
    for x, n in ((102.6, "1"), (130.5, "2"), (607.8, "1"), (635.8, "2")):
        label(ax, x, 286, "τ", 22, BROWN, italic=True)
        label(ax, x + 10.3, 290, n, 16.6, BROWN)
    label(ax, 548.4, 284.2, "τ", 22, BROWN, italic=True)
    label(ax, 558.7, 288.2, "H", 16.6, BROWN, italic=True)
    label(ax, 571.6, 292.1, "t", 15.1, BROWN, italic=True)
    label(ax, 295.5, 285.0, "…", 21, BROWN)
    label(ax, 243.4, 316.3, "Electricity Market", 23.5, BROWN)
    pylon(Icon(ax, 113.5, 349.2, 56.7, 58.7))
    operator(Icon(ax, 251.0, 354.1, 53.6, 56.6))
    exchange(ax, 295.2, 328.4, 392.2, 18.3, ORANGE, "#f4b183")
    coalition(ax, lower=True)
    label(ax, 111.9, 434.6, "Market")
    label(ax, 250.8, 434.0, "VPPO")
    label(ax, 309.2, 445.6, "DERs")
    label(ax, 192.5, 373.6, "$", 32, bold=False)
    arrow(ax, (170.7, 381.4), (241.1, 381.4), NAVY, 6.8, 2.2)
    arrow(ax, (239.7, 398.1), (171.4, 398.1), NAVY, 6.8, 2.2)
    electricity(Icon(ax, 189.5, 401.7, 32.7, 32.7))
    return fig

if __name__ == "__main__":
    import argparse
    from pathlib import Path as FilePath
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=FilePath, default=FilePath(__file__).parent / "Figs/timeline_python.pdf")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.family": "serif", "font.serif":["Times New Roman", "STIXGeneral", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "pdf.fonttype":42, "font.size":8.5}):
        fig = market_timeline()
        fig.savefig(args.output)
        plt.close(fig)
