"""Vispy overlays for the StageMapWidget.

All coordinates in the vispy scene are stage coordinates in µm (y-up, no
inversion, unlike the Qt-graphics based WellPlateView).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from vispy import scene

if TYPE_CHECKING:
    from collections.abc import Sequence

    import useq
    from vispy.scene import Node

    ColorLike = str | tuple[float, float, float, float]

# number of segments used to approximate a circular well outline
_CIRCLE_SEGMENTS = 48

# draw orders (images added by StageViewer have order <= -1)
ORDER_PLATE = 10
ORDER_FOVS = 20
ORDER_TRAIL = 30
ORDER_MARKERS = 40
ORDER_LABELS = 50

# the overlays are drawn over each other, so alpha has to actually blend
BLEND_STATE = {
    "depth_test": False,
    "blend": True,
    "blend_func": ("src_alpha", "one_minus_src_alpha"),
}

# the calibrated plate is drawn in the theme's foreground color (see set_colors);
# this is only the fallback for a widget that never gets a palette
CALIBRATED_COLOR = "#999999"
UNCALIBRATED_COLOR = "#E6A23C"
POSITION_COLOR = "#00A3FF"
FIRST_POSITION_COLOR = "#FFFFFF"
TRAIL_COLOR = (0.0, 0.64, 1.0, 0.6)  # POSITION_COLOR with alpha


def _rotation_matrix(rotation: float | None) -> np.ndarray:
    """Return the 2x2 (counter-clockwise) rotation matrix of a plate plan."""
    theta = np.deg2rad(rotation or 0.0)
    cos_, sin_ = np.cos(theta), np.sin(theta)
    return np.array([[cos_, -sin_], [sin_, cos_]])


def well_centers(plan: useq.WellPlatePlan) -> np.ndarray:
    """Return an (N, 2) array of stage coordinates (µm) of all well centers."""
    return np.array([(pos.x, pos.y) for pos in plan.all_well_positions], dtype=float)


@dataclass(frozen=True)
class WellHit:
    """The well of a plate plan closest to a given stage position."""

    row: int
    col: int
    name: str
    inside: bool


def nearest_well(plan: useq.WellPlatePlan, x: float, y: float) -> WellHit:
    """Return the well of `plan` whose center is closest to stage position (x, y).

    Parameters
    ----------
    plan : useq.WellPlatePlan
        The (calibrated) plate plan.
    x, y : float
        Stage coordinates in µm.
    """
    plate = plan.plate
    spacing_x, spacing_y = (s * 1000 for s in plate.well_spacing)  # mm -> µm
    rot = _rotation_matrix(plan.rotation)
    a1x, a1y = plan.a1_center_xy

    # map the stage position into the (unrotated) plate-local frame, in which
    # well (row, col) sits at (col * spacing_x, -row * spacing_y)
    local = rot.T @ np.array([x - a1x, y - a1y])
    col = int(np.clip(round(local[0] / spacing_x), 0, plate.columns - 1))
    row = int(np.clip(round(-local[1] / spacing_y), 0, plate.rows - 1))

    # offset of the point from the well center, in the plate-local frame
    dx = local[0] - col * spacing_x
    dy = local[1] + row * spacing_y
    well_w, well_h = (s * 1000 for s in plate.well_size)
    if plate.circular_wells:
        inside = (dx / (well_w / 2)) ** 2 + (dy / (well_h / 2)) ** 2 <= 1
    else:
        inside = abs(dx) <= well_w / 2 and abs(dy) <= well_h / 2

    name = str(plate.all_well_names[row, col])
    return WellHit(row, col, name, bool(inside))


class WellPlateOverlay:
    """Well outlines, plate boundary and well labels drawn in stage coordinates."""

    def __init__(self, parent: Node, face: str = "OpenSans") -> None:
        self._outlines = scene.visuals.Line(parent=parent, width=1)
        self._outlines.order = ORDER_PLATE
        self._border = scene.visuals.Line(parent=parent, width=1)
        self._border.order = ORDER_PLATE
        self._labels = scene.visuals.Text(
            parent=parent, font_size=8, anchor_x="center", anchor_y="center", face=face
        )
        self._labels.order = ORDER_LABELS
        for vis in (self._outlines, self._border, self._labels):
            # blending must be spelled out: set_gl_state replaces the visual's
            # default state, and without it the alpha of the colors is ignored
            vis.set_gl_state(**BLEND_STATE)
            vis.visible = False

        self._show_labels = True
        self._has_labels = False
        self._outline_color: ColorLike = CALIBRATED_COLOR
        self._label_color: ColorLike = CALIBRATED_COLOR
        self._calibrated = True

    # ----------------------------- public API -----------------------------

    @property
    def bound_visuals(self) -> tuple[scene.visuals.Line, ...]:
        """Visuals to consider when computing the scene bounds."""
        return (self._outlines, self._border) if self._outlines.visible else ()

    def set_labels_visible(self, visible: bool) -> None:
        """Show/hide the well name labels."""
        self._show_labels = visible
        self._labels.visible = visible and self._has_labels

    def set_colors(self, outline: ColorLike, label: ColorLike) -> None:
        """Set the colors used for a calibrated plate (uncalibrated stays amber)."""
        self._outline_color = outline
        self._label_color = label
        if self._calibrated:
            color = outline
            self._outlines.set_data(color=color)
            self._border.set_data(color=color)
            if self._has_labels:
                self._labels.color = label

    def set_label_font_size(self, size: float) -> None:
        """Scale the well labels (they are sized relative to the wells)."""
        if size != self._labels.font_size:
            self._labels.font_size = size

    def clear(self) -> None:
        """Remove the plate from the scene."""
        for vis in (self._outlines, self._border, self._labels):
            vis.visible = False
        self._has_labels = False

    def set_plan(self, plan: useq.WellPlatePlan | None, *, calibrated: bool) -> None:
        """Draw the wells of `plan` (or clear the overlay if None)."""
        if plan is None:
            self.clear()
            return

        self._calibrated = calibrated
        color = self._outline_color if calibrated else UNCALIBRATED_COLOR
        label_color = self._label_color if calibrated else UNCALIBRATED_COLOR
        centers = well_centers(plan)
        rot = _rotation_matrix(plan.rotation)

        outline_pts, outline_connect = self._well_outlines(plan, centers, rot)
        self._outlines.set_data(pos=outline_pts, connect=outline_connect, color=color)
        self._outlines.visible = True

        border_pts = self._plate_border(plan, rot)
        self._border.set_data(pos=border_pts, connect="strip", color=color)
        self._border.visible = True

        # well labels become unreadable (and expensive) on very dense plates
        names = [str(pos.name) for pos in plan.all_well_positions]
        self._has_labels = len(names) <= 384
        if self._has_labels:
            self._labels.text = names
            self._labels.pos = centers
            self._labels.color = label_color
        self._labels.visible = self._show_labels and self._has_labels

    # ----------------------------- helpers -----------------------------

    @staticmethod
    def _well_outlines(
        plan: useq.WellPlatePlan, centers: np.ndarray, rot: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (points, connect) drawing every well outline in one visual."""
        well_w, well_h = (s * 1000 for s in plan.plate.well_size)  # mm -> µm
        if plan.plate.circular_wells:
            theta = np.linspace(0, 2 * np.pi, _CIRCLE_SEGMENTS, endpoint=False)
            ring = np.column_stack(
                [well_w / 2 * np.cos(theta), well_h / 2 * np.sin(theta)]
            )
            ring = np.vstack([ring, ring[:1]])  # close the ring
        else:
            # (rotated) rectangle corners, closed
            half = np.array(
                [
                    (-well_w / 2, -well_h / 2),
                    (well_w / 2, -well_h / 2),
                    (well_w / 2, well_h / 2),
                    (-well_w / 2, well_h / 2),
                    (-well_w / 2, -well_h / 2),
                ]
            )
            ring = half @ rot.T

        n_ring = len(ring)
        pts = (centers[:, None, :] + ring[None, :, :]).reshape(-1, 2)
        connect = np.tile(
            np.append(np.ones(n_ring - 1, dtype=bool), False), len(centers)
        )
        return pts, connect

    @staticmethod
    def _plate_border(plan: useq.WellPlatePlan, rot: np.ndarray) -> np.ndarray:
        """Return the points of the plate boundary, with a chamfer near A1."""
        plate = plan.plate
        spacing_x, spacing_y = (s * 1000 for s in plate.well_spacing)
        # extent of the well centers in the plate-local frame
        width = (plate.columns - 1) * spacing_x
        height = (plate.rows - 1) * spacing_y
        mx, my = 0.75 * spacing_x, 0.75 * spacing_y
        chamfer = 0.75 * min(spacing_x, spacing_y)
        # counter-clockwise, starting just after the chamfered A1 (top-left) corner
        local = np.array(
            [
                (-mx, my - chamfer),
                (-mx, -height - my),
                (width + mx, -height - my),
                (width + mx, my),
                (-mx + chamfer, my),
                (-mx, my - chamfer),
            ]
        )
        return np.asarray(local @ rot.T + np.asarray(plan.a1_center_xy))


class PositionsOverlay:
    """Stage positions (markers + FOV rectangles), travel path and name labels."""

    def __init__(self, parent: Node, face: str = "OpenSans") -> None:
        self._fovs = scene.visuals.Line(parent=parent, width=1)
        self._fovs.order = ORDER_FOVS
        self._trail = scene.visuals.Arrow(
            parent=parent,
            connect="strip",
            width=1.5,
            arrow_type="stealth",
            arrow_size=8,
            arrow_color=TRAIL_COLOR,
        )
        self._trail.order = ORDER_TRAIL
        # note: vispy Markers ignores empty data (leaving its internal buffers
        # unset, which crashes bounds computations on camera.set_range), so an
        # invisible dummy point is used whenever there are no positions
        self._markers = scene.visuals.Markers(
            parent=parent, scaling="fixed", pos=np.zeros((1, 2)), size=0
        )
        self._markers.order = ORDER_MARKERS
        self._labels = scene.visuals.Text(
            parent=parent, font_size=9, anchor_x="center", anchor_y="top", face=face
        )
        self._labels.order = ORDER_LABELS
        for vis in (self._fovs, self._trail, self._markers, self._labels):
            vis.set_gl_state(**BLEND_STATE)
            vis.visible = False

        self._show_trail = True
        self._show_labels = True
        self._n_positions = 0

    # ----------------------------- public API -----------------------------

    @property
    def bound_visuals(self) -> tuple[scene.visuals.Markers, ...]:
        """Visuals to consider when computing the scene bounds."""
        return (self._markers,) if self._markers.visible else ()

    def set_trail_visible(self, visible: bool) -> None:
        """Show/hide the travel path between positions."""
        self._show_trail = visible
        self._trail.visible = visible and self._n_positions > 1

    def set_labels_visible(self, visible: bool) -> None:
        """Show/hide the position name labels."""
        self._show_labels = visible
        self._labels.visible = visible and self._n_positions > 0

    def set_positions(
        self,
        xy: Sequence[tuple[float, float]] | np.ndarray,
        names: Sequence[str],
        fov_size: tuple[float, float] | None,
    ) -> None:
        """Draw `xy` stage positions (µm), in travel order.

        Parameters
        ----------
        xy : sequence of (x, y)
            Stage coordinates of the positions, in µm.
        names : sequence of str
            One label per position (may be empty strings).
        fov_size : (width, height) | None
            Camera field of view in µm, drawn as a rectangle around each
            position. If None, no FOV rectangles are drawn.
        """
        pts = np.asarray(xy, dtype=float).reshape(-1, 2)
        self._n_positions = n = len(pts)
        if not n:
            self._markers.set_data(pos=np.zeros((1, 2)), size=0)
            for vis in (self._fovs, self._trail, self._markers, self._labels):
                vis.visible = False
            return

        # markers: first position highlighted (start of the travel path)
        edge_color = [FIRST_POSITION_COLOR] + [POSITION_COLOR] * (n - 1)
        self._markers.set_data(
            pos=pts,
            size=9,
            face_color=POSITION_COLOR,
            edge_color=edge_color,
            edge_width=1.5,
        )
        self._markers.visible = True

        # travel path with direction arrows at each segment end
        if n > 1:
            arrows = np.hstack([pts[:-1], pts[1:]])
            self._trail.set_data(
                pos=pts, connect="strip", color=TRAIL_COLOR, width=1.5, arrows=arrows
            )
        self._trail.visible = self._show_trail and n > 1

        # FOV rectangles (scene-scaled, so they naturally appear when zooming in)
        if fov_size is not None and fov_size[0] > 0 and fov_size[1] > 0:
            half = np.array(
                [
                    (-fov_size[0] / 2, -fov_size[1] / 2),
                    (fov_size[0] / 2, -fov_size[1] / 2),
                    (fov_size[0] / 2, fov_size[1] / 2),
                    (-fov_size[0] / 2, fov_size[1] / 2),
                    (-fov_size[0] / 2, -fov_size[1] / 2),
                ]
            )
            rect_pts = (pts[:, None, :] + half[None, :, :]).reshape(-1, 2)
            connect = np.tile(np.array([True] * 4 + [False]), n)
            self._fovs.set_data(pos=rect_pts, connect=connect, color=POSITION_COLOR)
            self._fovs.visible = True
        else:
            self._fovs.visible = False

        # labels slightly below each marker (well ids sit at the well centers)
        self._labels.text = [str(x) for x in names] or [""] * n
        self._labels.pos = pts
        self._labels.color = POSITION_COLOR
        self._labels.visible = self._show_labels
