from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING, cast

import cmap
import numpy as np
import vispy
import vispy.scene
import vispy.visuals
from qtpy.QtCore import QEvent, QObject, Qt, QTimer, Signal
from qtpy.QtWidgets import QLabel, QVBoxLayout, QWidget
from vispy import scene
from vispy.scene.visuals import Image

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from PyQt6.QtGui import QMouseEvent
    from vispy.app.canvas import MouseEvent
    from vispy.scene.widgets import ViewBox

    class VisualNode(vispy.scene.Node, vispy.visuals.Visual): ...


class StageViewer(QWidget):
    """A widget to add images with a transform to a vispy canves."""

    # Emitted around a Qt reparent (e.g. floating/redocking a dock widget that
    # contains this viewer) that recreates the underlying QOpenGLWidget's GL
    # context. vispy has no context-loss recovery: afterward the old visuals are
    # bound to invalid GL objects. On ``glContextAboutToReset`` owners should
    # hide/stop drawing their visuals; on ``glContextReset`` they should rebuild
    # them (construct fresh visuals so new GL objects are created on the live
    # context). This viewer rebuilds its own grid lines; owners rebuild theirs.
    glContextAboutToReset = Signal()
    glContextReset = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Stage Explorer")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._clims: tuple[float, float] | None = None
        self._cmap: cmap.Colormap = cmap.Colormap("gray")

        self.canvas = vispy.scene.SceneCanvas(show=True)

        self.view = cast("ViewBox", self.canvas.central_widget.add_view())
        self.view.camera = scene.PanZoomCamera(aspect=1)

        self._grid_visible = False
        self._create_grid()

        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.canvas.native)

        # Watch the native GL widget for the reparent that recreates its GL
        # context (napari does not forward hide/show to a floated dock's
        # content, so we watch the canvas widget itself, not this QWidget).
        self._gl_reset_pending = False
        self.canvas.native.installEventFilter(self)

        self._show_hover_label = True
        self._hover_pos_label = QLabel(self)
        self._hover_pos_label.setStyleSheet("color: rgba(100, 255, 255, 100); ")
        self._hover_pos_label.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents, True
        )
        self.canvas.events.mouse_move.connect(self._on_mouse_move)

    # --------------------GL CONTEXT RESET (reparent)--------------------

    def _create_grid(self) -> None:
        self._grid_lines = vispy.scene.GridLines(
            parent=self.view.scene, color="#888888", border_width=1
        )
        self._grid_lines.visible = self._grid_visible

    def eventFilter(self, obj: QObject | None, event: QEvent | None) -> bool:
        """Detect the QOpenGLWidget context reset caused by a Qt reparent."""
        if obj is self.canvas.native and event is not None:
            et = event.type()
            if et == QEvent.Type.Hide:
                self._suspend_gl()
            elif et in (QEvent.Type.Show, QEvent.Type.WinIdChange):
                # Hide visuals NOW (synchronously, before the imminent paintGL
                # draws them against dead handles), even if no Hide was seen.
                self._suspend_gl()
                if not self._gl_reset_pending:
                    # Rebuild deferred: the new context is created lazily on the
                    # next paintGL, so fresh CREATE commands must wait for it.
                    self._gl_reset_pending = True
                    QTimer.singleShot(0, self._do_gl_reset)
        return super().eventFilter(obj, event)

    def _suspend_gl(self) -> None:
        with suppress(Exception):
            self._grid_lines.visible = False
        self.glContextAboutToReset.emit()

    def _do_gl_reset(self) -> None:
        self._gl_reset_pending = False
        # rebuild our own grid (fresh gloo objects on the new context)...
        with suppress(Exception):
            self._grid_lines.parent = None
        self._create_grid()
        # ...then let owners rebuild their overlays/markers.
        self.glContextReset.emit()

    # --------------------PUBLIC METHODS--------------------

    def set_clims(self, clim: tuple[float, float] | None) -> None:
        """Set the color limits of the images in the scene."""
        self._clims = clim
        value = "auto" if clim is None else clim
        for child in self._get_images():
            child.clim = value

    def set_grid_visible(self, visible: bool) -> None:
        self._grid_visible = visible
        self._grid_lines.visible = visible

    def add_image(self, img: np.ndarray, transform: np.ndarray | None = None) -> None:
        """Add an image to the scene with the given transform.

        Parameters
        ----------
        img : np.ndarray
            The image to add to the scene. It should be a (Y, X) or (Y, X, 3) array.
        transform : np.ndarray | None
            The transform to apply to the image. It should be a 4x4 matrix.
            If None, the image will be added with the identity transform.
            The transformation is indented to be calculated elsewhere (in higher level
            widgets) based on, e.g., the stage position, pixel size, configuration
            affine, etc.  This is a relatively low-level, direct function.
        """
        # normalize the transform
        if transform is None:
            transform = np.eye(4)
        else:
            transform = np.asarray(transform)
            if transform.shape != (4, 4):
                raise ValueError("Transform must be a 4x4 matrix.")
            # vispy uses a column-major order for the transform matrix
            # so we need to transpose it to get the correct order
            if np.allclose(transform[-1], (0, 0, 0, 1)):
                transform = transform.T

        # add the image to the scene with the transform
        # texture_format="auto" uses GPUScaledTexture2D so that clim changes
        # only update a shader uniform instead of re-uploading the texture.
        frame = Image(
            img,
            cmap=self._cmap.to_vispy(),
            parent=self.view.scene,
            clim="auto" if self._clims is None else self._clims,
            texture_format="auto",
        )
        # keep the added image on top of the others
        frame.order = min(child.order for child in self._get_images()) - 1
        frame.transform = scene.MatrixTransform(matrix=transform)

    def clear(self) -> None:
        """Clear the scene."""
        for child in reversed(self.view.scene.children):
            if isinstance(child, Image):
                child.parent = None

    def zoom_to_fit(self, *, margin: float = 0.05) -> None:
        """Recenter the view to the center of all images.

        Parameters
        ----------
        margin : float
            Extra margin to add between the images and the edge of the view.
            This is a percentage of the view size. Default is 0.05 (5%).
        """
        if not (visuals := self._get_images()):
            return
        x_bounds, y_bounds, *_ = get_vispy_scene_bounds(visuals)
        self.view.camera.set_range(x=x_bounds, y=y_bounds, margin=margin)

    def canvas_to_world(self, canvas_pos: tuple[float, float]) -> tuple[float, float]:
        """Convert canvas coordinates to world coordinates."""
        # map canvas position to world position
        world_x, world_y, *_ = self.view.scene.transform.imap(canvas_pos)
        return world_x, world_y

    def world_to_canvas(self, world_pos: tuple[float, float]) -> tuple[float, float]:
        """Convert world coordinates to canvas coordinates."""
        # map world position to canvas position
        canvas_x, canvas_y, *_ = self.view.scene.transform.map(world_pos)
        return canvas_x, canvas_y

    # --------------------PRIVATE METHODS--------------------

    def _get_images(self) -> Iterator[Image]:
        """Yield images in the scene."""
        for child in self.view.scene.children:
            if isinstance(child, Image):
                yield child

    def _on_mouse_move(self, event: MouseEvent) -> None:
        if not self._show_hover_label:
            return  # pragma: no cover

        # map canvas position to world position
        world_x, world_y = self.canvas_to_world(event.pos)
        self._hover_pos_label.setText(f"({world_x:.2f}, {world_y:.2f})")
        self._hover_pos_label.adjustSize()

        # move hover label to the mouse position
        # ensure horizontally and vertically within the view
        lbl_width = self._hover_pos_label.width()
        x = event.pos[0] - (lbl_width // 2)
        margin = 5
        x = max(margin, min(x, self.width() - lbl_width - margin))
        y = event.pos[1] - 32
        if y < 8:
            y += 48
        self._hover_pos_label.move(x, y)
        self._hover_pos_label.setVisible(True)

    def leaveEvent(self, a0: QMouseEvent | None) -> None:
        self._hover_pos_label.setVisible(False)  # pragma: no cover


def get_vispy_scene_bounds(
    visuals: Iterable[VisualNode],
) -> tuple[list[float], list[float], list[float]]:
    """Get the bounding box for `visuals` in world coordinates."""
    # tracks: [xmin, xmax], [ymin, ymax], [zmin, zmax]
    bounds = np.array([[np.inf, -np.inf], [np.inf, -np.inf], [np.inf, -np.inf]])

    for obj in visuals:
        (x_min, x_max), (y_min, y_max) = obj.bounds(0), obj.bounds(1)
        local_bounds = np.array([[x_min, y_min, 0, 1], [x_max, y_max, 0, 1]])

        # Map local bounds to world coordinates
        transform = obj.node_transform(obj.scene_node)
        world_bounds = transform.map(local_bounds)

        # Convert from homogeneous to 3D coordinates
        world_bounds = world_bounds[:, :3] / world_bounds[:, 3, np.newaxis]

        # Update world bounds
        bounds[:, 0] = np.minimum(bounds[:, 0], world_bounds.min(axis=0))
        bounds[:, 1] = np.maximum(bounds[:, 1], world_bounds.max(axis=0))

    # replace inf values with 0 ... better than -inf
    bounds = np.where(np.isinf(bounds), 0, bounds)

    return tuple(bounds.tolist())
