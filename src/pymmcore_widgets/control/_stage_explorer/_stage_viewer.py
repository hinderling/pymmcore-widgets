from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING, cast

import cmap
import numpy as np
import vispy
import vispy.scene
import vispy.visuals
from qtpy.QtCore import QEvent, QObject, Qt, QTimer, Signal
from qtpy.QtGui import QPalette
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
    # context. vispy has no context-loss recovery: afterward the old visuals
    # are bound to invalid GL objects, and flushing their stale commands
    # (draws OR deletes) into the window's SHARED GL namespace can corrupt
    # every other canvas in the application (e.g. the napari viewer). On a
    # true reparent this viewer therefore replaces the ENTIRE canvas: the old
    # one is parked unpainted forever and `self.canvas`/`self.view` point to a
    # fresh one. On ``glContextAboutToReset`` owners must hide/stop drawing
    # their visuals; on ``glContextReset`` they must rebuild their whole scene
    # against the (possibly new) ``self.view.scene`` and re-connect anything
    # they had connected to ``self.canvas.events``.
    glContextAboutToReset = Signal()
    glContextReset = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Stage Explorer")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._clims: tuple[float, float] | None = None
        self._cmap: cmap.Colormap = cmap.Colormap("gray")
        self._grid_visible = False

        # reparent bookkeeping (see eventFilter / _on_context_destroyed)
        self._gl_reset_pending = False
        self._needs_canvas_recreate = False
        self._gl_context = None  # QOpenGLContext tracked at each paint
        # canvases replaced after a reparent. Kept referenced and unpainted:
        # if one were garbage collected, vispy would queue GL DELETEs for
        # handles of a destroyed context, and a later flush could run them in
        # the shared namespace where the ids now belong to live objects.
        self._dead_canvases: list[vispy.scene.SceneCanvas] = []

        self._create_canvas()

        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.canvas.native)

        self._show_hover_label = True
        self._hover_pos_label = QLabel(self)
        self._style_hover_label()
        self._hover_pos_label.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents, True
        )

    # --------------------GL CONTEXT RESET (reparent)--------------------

    def _create_canvas(self) -> None:
        """Create the vispy canvas, view, camera and grid (fresh GL context)."""
        self.canvas = vispy.scene.SceneCanvas(show=True)
        self.view = cast("ViewBox", self.canvas.central_widget.add_view())
        self.view.camera = scene.PanZoomCamera(aspect=1)
        self._create_grid()
        # Watch the native GL widget: hide/show around reparents (napari does
        # not forward hide/show to a floated dock's content, so we watch the
        # canvas widget itself, not this QWidget) and paints to track the live
        # QOpenGLContext (see eventFilter).
        self.canvas.native.installEventFilter(self)
        self.canvas.events.mouse_move.connect(self._on_mouse_move)

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
            elif et == QEvent.Type.Show:
                # Hide visuals NOW (synchronously, before the imminent paintGL
                # draws them against possibly dead handles) and rebuild them.
                self._suspend_gl()
                self._schedule_reset()
            elif et == QEvent.Type.Paint:
                # Track the context vispy is flushing into. Its
                # aboutToBeDestroyed signal is the ground truth for context
                # loss: the Qt events delivered around a dock reparent vary
                # with native-realization timing (WinIdChange may never come),
                # but the context always announces its own destruction.
                self._track_context(getattr(obj, "context", lambda: None)())
        return super().eventFilter(obj, event)

    def _track_context(self, ctx: QObject | None) -> None:
        if ctx is None or ctx is self._gl_context:
            return
        self._untrack_context()
        self._gl_context = ctx
        ctx.aboutToBeDestroyed.connect(self._on_context_destroyed)

    def _untrack_context(self) -> None:
        if self._gl_context is not None:
            with suppress(Exception):
                self._gl_context.aboutToBeDestroyed.disconnect(
                    self._on_context_destroyed
                )
            self._gl_context = None

    def _on_context_destroyed(self) -> None:
        """The native's GL context is being destroyed (e.g. a dock reparent).

        Everything vispy uploaded is gone; this canvas can never be drawn
        again (a later paint would flush stale commands into a lazily created
        context in the shared namespace). Stop its paints immediately and
        replace it.
        """
        self._untrack_context()
        with suppress(Exception):
            self.canvas.native.setUpdatesEnabled(False)
        self._needs_canvas_recreate = True
        self._suspend_gl()
        self._schedule_reset()

    def _schedule_reset(self) -> None:
        if not self._gl_reset_pending:
            # Deferred: the new context is created lazily on the next paintGL,
            # so fresh CREATE commands must wait for it.
            self._gl_reset_pending = True
            QTimer.singleShot(0, self._do_gl_reset)

    def _suspend_gl(self) -> None:
        with suppress(Exception):
            self._grid_lines.visible = False
        self.glContextAboutToReset.emit()

    def _do_gl_reset(self) -> None:
        self._gl_reset_pending = False
        try:
            if self._needs_canvas_recreate:
                self._needs_canvas_recreate = False
                self._recreate_canvas()
            else:
                # same context (plain hide/show): fresh grid on the live canvas
                with suppress(Exception):
                    self._grid_lines.parent = None
                self._create_grid()
        except RuntimeError:
            # the widget is being torn down (context destruction at shutdown
            # also schedules a reset); nothing left to rebuild
            return
        # let owners rebuild their overlays/markers on self.view.scene
        self.glContextReset.emit()

    def _recreate_canvas(self) -> None:
        """Replace the whole canvas after a reparent destroyed its GL context.

        The old canvas is unsalvageable and even deleting it is unsafe (its
        teardown would flush stale GL handles into the shared namespace), so
        it is parked: hidden, updates disabled, never painted again, and kept
        referenced in ``_dead_canvases``. Any image visuals owned by the old
        scene are gone; owners repopulate via the ``glContextReset`` signal.
        """
        old = self.canvas
        self._untrack_context()  # no-op if the destroy signal already did
        with suppress(Exception):
            old.native.removeEventFilter(self)
        with suppress(Exception):
            old.events.mouse_move.disconnect(self._on_mouse_move)
        old.native.setUpdatesEnabled(False)
        old.native.hide()
        if (layout := self.layout()) is not None:
            layout.removeWidget(old.native)
        old.native.setParent(None)
        self._dead_canvases.append(old)

        rect = self.view.camera.rect
        self._create_canvas()
        if (layout := self.layout()) is not None:
            layout.addWidget(self.canvas.native)
        self.canvas.native.show()
        self.view.camera.rect = rect  # keep the zoom/pan across the swap

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

    def _style_hover_label(self) -> None:
        """Color the hover position readout from the palette.

        The theme's own foreground color stays readable on light and dark
        themes alike.
        """
        fg = self.palette().color(QPalette.ColorRole.WindowText)
        self._hover_pos_label.setStyleSheet(
            f"color: rgba({fg.red()}, {fg.green()}, {fg.blue()}, 200);"
        )

    def changeEvent(self, event: QEvent | None) -> None:
        """Follow the application theme when it changes."""
        super().changeEvent(event)
        if event is not None and event.type() in (
            QEvent.Type.PaletteChange,
            QEvent.Type.StyleChange,
        ):
            self._style_hover_label()

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
