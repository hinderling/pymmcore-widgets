from __future__ import annotations

import re
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import useq
from pymmcore_plus import CMMCorePlus
from qtpy.QtCore import QEvent, QItemSelectionModel, QSize, Qt, Signal
from qtpy.QtGui import QFontInfo, QPalette
from qtpy.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QSizePolicy,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from superqt import QIconifyIcon
from superqt.utils import signals_blocked
from vispy.scene import MatrixTransform, Node

from pymmcore_widgets.control._q_stage_controller import QStageMoveAccumulator
from pymmcore_widgets.control._rois.roi_manager import GRAY
from pymmcore_widgets.control._stage_explorer._stage_explorer import (
    AffineState,
    _StagePoller,
)
from pymmcore_widgets.control._stage_explorer._stage_position_marker import (
    StagePositionMarker,
)
from pymmcore_widgets.control._stage_explorer._stage_viewer import (
    StageViewer,
    get_vispy_scene_bounds,
)
from pymmcore_widgets.useq_widgets._positions import PositionTable, well_id
from pymmcore_widgets.useq_widgets._well_plate_widget import _sort_plate

from ._calibration_store import PlateCalibrationStore
from ._overlays import (
    SELECTED_COLOR_DARK_BG,
    SELECTED_COLOR_LIGHT_BG,
    PositionsOverlay,
    WellPlateOverlay,
    nearest_well,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from PyQt6.QtGui import QAction
    from qtpy.QtGui import QCloseEvent, QColor, QIcon, QShowEvent
    from vispy.app.canvas import MouseEvent

    from pymmcore_widgets.control._stage_explorer._stage_viewer import VisualNode
else:
    from qtpy.QtWidgets import QAction

NO_PLATE = "None"
# icon/status colors, matching the palette used by the other widgets
GREEN = "#3A3"
RED = "#C33"
ORANGE = "#E6A23C"


def _ui_font_face() -> str:
    """Return a font family for the canvas labels that matches the rest of the UI.

    The Qt default resolves to a private alias on some platforms (".AppleSystemUI
    Font"), which the canvas font loader cannot open, so fall back to a family it
    can find.
    """
    if QApplication.instance() is not None:
        family = QFontInfo(QApplication.font()).family()
        if family and not family.startswith("."):
            return str(family)
    return "Helvetica"


def _rgba(color: QColor, alpha: float = 1.0) -> tuple[float, float, float, float]:
    """Return a vispy style rgba tuple for a Qt color."""
    return (color.redF(), color.greenF(), color.blueF(), alpha)


# parents that say nothing about who owns a position table
_ANONYMOUS_PARENTS = ("QWidget", "QStackedWidget", "QTabWidget", "QScrollArea")


def _table_title(table: PositionTable) -> str:
    """Return a human readable name for a position table, based on its owner."""
    # walk up to a meaningful ancestor: the MDA widget, a wizard, a window...
    parent = table.parentWidget()
    while parent is not None:
        if title := (parent.windowTitle() or ""):
            return f"{title} positions"
        name = type(parent).__name__
        if name not in _ANONYMOUS_PARENTS and not name.endswith("Tabs"):
            return f"{_split_camel(name)} positions"
        parent = parent.parentWidget()
    return _split_camel(type(table).__name__)


def _split_camel(name: str) -> str:
    """Turn "MDAWidget" into "MDA Widget"."""
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", name)


class StageMapWidget(QWidget):
    """A live map of the XY stage with well plate overlay and stored positions.

    This widget shows a zoomable/pannable "bird's eye" view of the stage in stage
    coordinates: the live stage position, an (optionally calibrated) well plate
    outline, and the positions of an existing position list, connected by a
    travel path showing the order in which they will be visited.
    Double-clicking anywhere on the map moves the stage to that position (without
    blocking the GUI).

    The widget deliberately keeps no position list of its own: it binds to a
    [PositionTable][pymmcore_widgets.PositionTable] that already exists in the
    application, typically the one inside an
    [MDAWidget][pymmcore_widgets.MDAWidget], so there is never a second list to
    keep in sync.  The first table found is bound automatically when the widget
    is shown; the toolbar lets the user pick another (see `setPositionTable`).

    Plate calibrations (the stage coordinates of well A1 and the plate rotation)
    are persisted to disk, one file per plate format, and automatically reloaded
    the next time the same plate format is selected, so a plate only needs to be
    calibrated once per stage/holder, not once per experiment.  Calibrations can
    be redone, imported/exported, or deleted at any time from the toolbar.

    With a calibrated plate, the "Assign Positions to Wells" action fills in the
    well of each position in the bound table (the closest well centroid), which
    useq stores as ``plate_row``/``plate_col`` and the OME-Zarr writers turn
    into the plate/well hierarchy of the dataset.  Position names are left
    alone, so they stay free for your own labels.

    Parameters
    ----------
    parent : QWidget | None
        Optional parent widget, by default None.
    mmcore : CMMCorePlus | None
        Optional [`CMMCorePlus`][pymmcore_plus.CMMCorePlus] micromanager core.
        By default, None. If not specified, the widget will use the active
        (or create a new)
        [`CMMCorePlus.instance`][pymmcore_plus.core._mmcore_plus.CMMCorePlus.instance].
    calibration_dir : Path | str | None
        Optional directory in which plate calibrations are persisted.  By
        default, a "pymmcore-widgets/plate_calibrations" folder in the user's
        standard data location.
    """

    platePlanChanged = Signal(object)  # useq.WellPlatePlan | None
    positionTableChanged = Signal(object)  # PositionTable | None
    wellsAssigned = Signal()

    def __init__(
        self,
        parent: QWidget | None = None,
        mmcore: CMMCorePlus | None = None,
        *,
        calibration_dir: Path | str | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Stage Map")

        self._mmc = mmcore or CMMCorePlus.instance()
        self._store = PlateCalibrationStore(calibration_dir)

        # current plate plan (calibrated or drawn at the default A1=(0,0))
        self._plan: useq.WellPlatePlan | None = None
        self._calibrated: bool = False
        # plates set programmatically that are not in the useq registry
        self._custom_plates: dict[str, useq.WellPlate] = {}

        self._stage_controller: QStageMoveAccumulator | None = None
        # whether the poll toggle is on (the poller itself may be paused while
        # the widget is hidden, see hideEvent/showEvent)
        self._polling: bool = False
        self._poll_interval_ms: int = 33
        # True while the widget is hidden across a reparent (dock float/redock),
        # which recreates the underlying QOpenGLWidget's GL context and
        # invalidates every visual. While set, GL draws are skipped (drawing into
        # the recreated context is invalid); visuals are rebuilt once shown again.
        self._gl_suspended: bool = False
        # Latched when teardown begins. Guards every deferred/threaded callback so
        # none touch the canvas, core or poller while they are being destroyed.
        self._closing: bool = False
        # Poll the (potentially slow, serial) stage position on a BACKGROUND
        # thread so it never blocks map drawing / zoom / pan. It emits
        # positionChanged only when the stage actually moves; the marker + label
        # update on the GUI thread from that signal. There is NO GUI-side
        # timer: label fonts rescale from the canvas draw event, so an open
        # map costs the GUI thread nothing while it is not being interacted
        # with.
        self._last_stage_pos: tuple[float, float] | None = None
        self._stage_poller = _StagePoller(
            self._mmc, self, interval_ms=self._poll_interval_ms, poll_fov=True
        )
        self._stage_poller.positionChanged.connect(self._on_stage_position_polled)
        self._stage_poller.fovChanged.connect(self._on_fov_polled)
        # the position table this widget mirrors; this widget has none of its
        # own, see setPositionTable
        self._pos_table: PositionTable | None = None
        self._auto_bind_attempted: bool = False
        self._calib_dialog: _PlateCalibrationDialog | None = None
        # last field of view size used to draw the position rectangles
        self._last_fov_size: tuple[float, float] | None = None
        # stage-µm coordinates of the drawn positions (for click hit tests)
        self._pos_xy: np.ndarray | None = None
        # last zoom the label fonts were sized for; None forces a resize
        self._last_px_per_um: float | None = None
        # camera rect of the last zoom_to_fit; while the camera still matches
        # it, a canvas resize triggers a re-fit (see _on_canvas_resize)
        self._fit_rect: tuple[float, float, float, float] | None = None

        # WIDGETS ------------------------------------------------------------

        self._stage_viewer = StageViewer(self)
        self._stage_viewer.setCursor(Qt.CursorShape.CrossCursor)

        # A Qt reparent (dock float/redock) recreates the canvas GL context and
        # invalidates every visual; StageViewer detects it and signals us to
        # hide (about-to-reset) then rebuild (reset) our overlays + marker.
        self._stage_viewer.glContextAboutToReset.connect(self._on_gl_about_to_reset)
        self._stage_viewer.glContextReset.connect(self._on_gl_reset)
        # PLATE-SPACE rendering. Everything (plate, positions, marker) is drawn
        # in STAGE-µm coordinates under this node; its transform maps stage ->
        # canonical plate space so the plate ALWAYS reads A1 top-left and the
        # marker shows the true stage position on that fixed plate. Identity
        # until a plan is set (see _update_plate_transform).
        self._font_face = _ui_font_face()
        # Intended marker visibility (the poll toggle). Tracked separately so
        # a reparent, which hides then rebuilds the marker, can restore it.
        self._marker_visible: bool = False
        self._build_scene()

        # cached parameters for efficient affine calculations
        self._affine_state = AffineState(self._mmc)

        self._toolbar = tb = _StageMapToolbar(self)
        self._stage_pos_label = QLabel()
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        tb.addWidget(spacer)
        tb.addWidget(self._stage_pos_label)

        self._status_label = QLabel()
        self._status_label.setContentsMargins(6, 2, 6, 2)
        self._source_label = QLabel()
        self._source_label.setContentsMargins(6, 2, 6, 2)
        self._source_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self._source_label.setStyleSheet("color: palette(mid);")

        # LAYOUT -------------------------------------------------------------

        status_row = QHBoxLayout()
        status_row.setContentsMargins(0, 0, 0, 0)
        status_row.addWidget(self._status_label, 1)
        status_row.addWidget(self._source_label, 0)

        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(0)
        # a thin frame around the canvas, so the extent of the map stays visible
        # once its background matches the surrounding application
        self._map_frame = QFrame()
        # named, so the stylesheet below cannot leak into child widgets
        # (QLabel is itself a QFrame subclass)
        self._map_frame.setObjectName("stageMapFrame")
        frame_layout = QVBoxLayout(self._map_frame)
        frame_layout.setContentsMargins(1, 1, 1, 1)
        frame_layout.setSpacing(0)
        frame_layout.addWidget(self._stage_viewer)

        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self._toolbar, 0)
        main_layout.addWidget(self._map_frame, 1)
        main_layout.addLayout(status_row, 0)

        # CONNECTIONS ---------------------------------------------------------

        tb.plate_combo.currentTextChanged.connect(self._on_plate_combo_changed)
        tb.calibrate_action.triggered.connect(self.calibrate)
        tb.import_action.triggered.connect(self._import_calibration)
        tb.export_action.triggered.connect(self._export_calibration)
        tb.delete_action.triggered.connect(self._delete_calibration)
        tb.assign_action.triggered.connect(self.assign_wells_to_positions)
        # via a widget method, NOT the overlay's bound method: _build_scene
        # replaces the overlay object on every GL reset
        tb.trails_action.toggled.connect(self._on_trails_toggled)
        tb.labels_action.toggled.connect(self._on_labels_toggled)
        tb.poll_action.toggled.connect(self._on_poll_stage_toggled)
        tb.grid_action.toggled.connect(self._stage_viewer.set_grid_visible)
        tb.zoom_to_fit_action.triggered.connect(self.zoom_to_fit)
        tb.source_menu.aboutToShow.connect(self._populate_source_menu)

        self._mmc.events.systemConfigurationLoaded.connect(self._on_sys_config_loaded)
        self._mmc.events.pixelSizeChanged.connect(self._on_pixel_size_changed)
        self._mmc.events.roiSet.connect(self._on_roi_changed)
        self._mmc.events.propertyChanged.connect(self._on_property_changed)
        # commanded moves report through the core callback instantly; the
        # background poller only backstops moves the adapter never reports
        # (e.g. joystick on a serial stage)
        self._mmc.events.XYStagePositionChanged.connect(self._on_xy_event)

        self._connect_canvas_events()

        self.destroyed.connect(self._disconnect)

        # INITIALIZATION -------------------------------------------------------

        self._apply_palette()
        self._on_sys_config_loaded()
        self._update_status()
        self._update_source_label()
        self._update_action_enablement()
        tb.poll_action.setChecked(True)
        self.zoom_to_fit()

    # -----------------------------PUBLIC METHODS-------------------------------------

    def toolBar(self) -> QToolBar:
        """Return the toolbar of the widget."""
        return self._toolbar

    def positionTable(self) -> PositionTable | None:
        """Return the positions table the map currently reflects, if any."""
        return self._pos_table

    def setPositionTable(self, table: PositionTable | None) -> None:
        """Show the positions of `table` on the map.

        This widget deliberately has no position list of its own: it binds to an
        existing [PositionTable][pymmcore_widgets.PositionTable], typically the
        one inside an [MDAWidget][pymmcore_widgets.MDAWidget], so that the map
        shows (and can rename) the positions that will actually be acquired,
        with no second list to keep in sync.  Pass None to show no positions.

        If the table goes away, the map unbinds itself.  See also
        `linkablePositionTables`, which finds the tables that currently exist in
        the application, and `bindToFirstPositionTable`.
        """
        if table is self._pos_table:
            return

        # unhook the previous source
        if self._pos_table is not None:
            with suppress(RuntimeError, TypeError):
                self._pos_table.valueChanged.disconnect(self._refresh_positions)
                self._pos_table.destroyed.disconnect(self._on_table_destroyed)
                self._pos_table.table().itemSelectionChanged.disconnect(
                    self._sync_selection_from_table
                )

        self._pos_table = table
        if table is not None:
            table.valueChanged.connect(self._refresh_positions)
            table.destroyed.connect(self._on_table_destroyed)
            table.table().itemSelectionChanged.connect(
                self._sync_selection_from_table
            )

        self._update_source_label()
        self._update_action_enablement()
        self._refresh_positions()
        self.positionTableChanged.emit(table)

    def linkablePositionTables(self) -> dict[str, PositionTable]:
        """Return the position tables that currently exist in this application.

        Maps a human readable title (e.g. "MDA Widget positions") to the table.
        Intended for building a UI to choose between them; see
        `setPositionTable`.
        """
        found: dict[str, PositionTable] = {}
        app = QApplication.instance()
        if not isinstance(app, QApplication):  # pragma: no cover
            return found
        for top in app.topLevelWidgets():
            # a table can be a top-level widget itself (findChildren excludes it)
            tables = top.findChildren(PositionTable)
            if isinstance(top, PositionTable):
                tables = [top, *tables]
            for table in tables:
                title = _table_title(table)
                # disambiguate if several tables end up with the same title
                if title in found:  # pragma: no cover
                    title = f"{title} ({len(found)})"
                found[title] = table
        return found

    def bindToFirstPositionTable(self) -> PositionTable | None:
        """Bind to the first position table found in the application, if any.

        Returns the table that was bound, or None if none was found.  This is
        called automatically the first time the widget is shown, so that
        dropping it next to an MDA widget just works.
        """
        if (tables := self.linkablePositionTables()) and (
            table := next(iter(tables.values()))
        ):
            self.setPositionTable(table)
            return table
        return None

    def changeEvent(self, event: QEvent | None) -> None:
        """Follow the application theme when it changes (e.g. napari light/dark)."""
        super().changeEvent(event)
        if event is not None and event.type() in (
            QEvent.Type.PaletteChange,
            QEvent.Type.StyleChange,
        ):
            self._apply_palette()

    def _on_gl_about_to_reset(self) -> None:
        """A Qt reparent is recreating the GL context (StageViewer signal).

        Hide every visual: vispy draws them on each paintGL regardless of
        updates, and drawing into the recreated context is invalid until the
        visuals are rebuilt in _on_gl_reset.
        """
        if self._closing:
            return
        self._gl_suspended = True
        with suppress(Exception):
            self._stage_pos_marker.visible = False
            self._plate_overlay.hide()
            self._positions_overlay.hide()

    def _build_scene(self) -> None:
        """Create the plate node, overlays and marker on the CURRENT canvas."""
        self._plate_node = Node(parent=self._stage_viewer.view.scene)
        self._plate_node.transform = MatrixTransform()
        self._plate_overlay = WellPlateOverlay(self._plate_node, self._font_face)
        self._positions_overlay = PositionsOverlay(self._plate_node, self._font_face)
        self._build_stage_marker()

    def _connect_canvas_events(self) -> None:
        """(Re)connect to the CURRENT canvas's events.

        Called at construction and after every GL reset: on a reparent the
        StageViewer replaces the whole canvas, and connections made on the old
        one die with it.
        """
        canvas = self._stage_viewer.canvas
        canvas.events.mouse_double_click.connect(self._on_mouse_double_click)
        canvas.events.mouse_release.connect(self._on_mouse_release)
        # zoom/pan/resize all end in a draw; rescale the labels right there
        # instead of from a polling GUI timer (see _on_canvas_draw)
        canvas.events.draw.connect(self._on_canvas_draw)
        canvas.events.resize.connect(self._on_canvas_resize)

    def _on_gl_reset(self) -> None:
        """The GL context was reset (StageViewer signal, deferred).

        On a true reparent the StageViewer replaced the entire canvas, so the
        whole scene, plate node included, is rebuilt against the current
        ``view.scene`` and the canvas-event connections are renewed. On a
        plain hide/show the canvas survives; the old subtree is detached
        first, so rebuilding on it is just a fresh start, not a duplicate.
        """
        if self._closing:
            return
        with suppress(Exception):
            self._plate_node.parent = None  # drop the old subtree if any
        self._build_scene()
        self._connect_canvas_events()
        self._gl_suspended = False
        # repopulate the fresh visuals from the widget's source-of-truth state
        self._update_plate_transform()  # stage->plate node transform
        self._apply_palette()  # overlay colors + canvas background
        self._plate_overlay.set_plan(self._plan, calibrated=self._calibrated)
        labels_on = self._toolbar.labels_action.isChecked()
        self._plate_overlay.set_labels_visible(labels_on)
        self._positions_overlay.set_labels_visible(labels_on)
        self._positions_overlay.set_trail_visible(
            self._toolbar.trails_action.isChecked()
        )
        self._refresh_positions()
        if self._marker_visible and self._last_stage_pos is not None:
            x, y = self._last_stage_pos
            matrix = self._affine_state.system_affine_translated(x, y)
            self._stage_pos_marker.apply_transform(matrix.T)
        self._last_px_per_um = None  # fresh visuals: labels need resizing
        self._update_label_font_sizes()
        self._stage_viewer.canvas.update()

    def showEvent(self, event: QShowEvent | None) -> None:
        """Bind to an existing position table the first time we are shown."""
        super().showEvent(event)
        if self._pos_table is None and not self._auto_bind_attempted:
            self._auto_bind_attempted = True
            self.bindToFirstPositionTable()
        # resume background polling if the toggle is on (see hideEvent)
        if self._polling and not self._closing:
            with suppress(Exception):
                if not self._stage_poller.isRunning():
                    self._stage_poller.start()

    def hideEvent(self, event: QEvent | None) -> None:
        """Stop the background poller whenever the widget is hidden.

        A widget is hidden before it is destroyed, so this reliably stops the
        poller before its QThread would be torn down (a QThread still running
        at destruction aborts the process). It also covers dock float/redock;
        showEvent restarts polling if the toggle is still on.
        """
        with suppress(Exception):
            self._stage_poller.stop()
        super().hideEvent(event)

    def closeEvent(self, event: QCloseEvent | None) -> None:
        """Tear down on close in addition to on destroy.

        The `destroyed` signal is not always delivered to a Python slot when a
        dock's content widget is destroyed, so run teardown here too. It stops
        the poller before its QThread is destroyed; `_disconnect` is idempotent.
        """
        self._disconnect()
        super().closeEvent(event)

    @property
    def poll_stage_position(self) -> bool:
        """Whether the live stage position is being polled and displayed."""
        return self._polling

    @poll_stage_position.setter
    def poll_stage_position(self, value: bool) -> None:
        self._toolbar.poll_action.setChecked(value)

    def platePlan(self) -> useq.WellPlatePlan | None:
        """Return the current *calibrated* plate plan, or None.

        Returns None both when no plate is selected and when the selected plate
        has not been calibrated yet (the overlay is then drawn with A1 at
        (0, 0) as a preview).
        """
        return self._plan if self._calibrated else None

    def setPlatePlan(
        self, value: str | useq.WellPlate | useq.WellPlatePlan | None
    ) -> None:
        """Set the plate overlay.

        Parameters
        ----------
        value : str | useq.WellPlate | useq.WellPlatePlan | None
            If a string, the name of a registered plate (e.g. "96-well"); if a
            [useq.WellPlate][], the plate to overlay. In both cases a stored
            calibration is loaded if available.  If a [useq.WellPlatePlan][],
            the plan is used as-is and considered calibrated (but not
            persisted).  If None, the overlay is removed.
        """
        if value is None:
            self._select_combo_name(NO_PLATE)
            self._set_plan(None, calibrated=False)
            return

        if isinstance(value, useq.WellPlatePlan):
            plate = value.plate
            self._remember_plate(plate)
            self._select_combo_name(plate.name)
            self._set_plan(value, calibrated=True)
            return

        if isinstance(value, str):
            plate = useq.WellPlate.from_str(value)
        else:
            plate = value
            self._remember_plate(plate)

        self._select_combo_name(plate.name)
        self._apply_plate(plate)

    def zoom_to_fit(self, *, margin: float = 0.05) -> None:
        """Zoom the view to everything that gives the map its extent.

        That is the plate and the positions, plus the stage marker so the
        current position is never off screen.  With nothing but the marker, the
        view is framed around it at a few fields of view rather than filling
        the canvas with a single field.
        """
        visuals: list[VisualNode] = [
            *self._plate_overlay.bound_visuals,
            *self._positions_overlay.bound_visuals,
        ]
        if self._stage_pos_marker.visible:
            visuals.append(self._stage_pos_marker)
        if not visuals:
            return

        (x0, x1), (y0, y1), *_ = get_vispy_scene_bounds(visuals)
        # never zoom in further than a few fields of view: fitting a lone marker
        # would otherwise slam the view onto a single field
        fov = self._last_fov_size or self._fov_size() or (200.0, 200.0)
        min_w, min_h = fov[0] * 4, fov[1] * 4
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if (x1 - x0) < min_w:
            x0, x1 = cx - min_w / 2, cx + min_w / 2
        if (y1 - y0) < min_h:
            y0, y1 = cy - min_h / 2, cy + min_h / 2

        self._stage_viewer.view.camera.set_range(x=(x0, x1), y=(y0, y1), margin=margin)
        # remember the fitted rect: while the camera still matches it (the
        # user has not zoomed/panned), a canvas resize re-fits automatically
        r = self._stage_viewer.view.camera.rect
        self._fit_rect = (r.left, r.bottom, r.width, r.height)

    def value(
        self, exclude_unchecked: bool = True, exclude_hidden_cols: bool = True
    ) -> tuple[useq.Position, ...] | useq.WellPlatePlan:
        """Return the positions currently shown on the map.

        These come from the bound position table (see `setPositionTable`); an
        empty tuple is returned if no table is bound.  If a plate is calibrated
        and positions have been assigned to wells (see
        `assign_wells_to_positions`), each position carries its
        `plate_row`/`plate_col` indices.
        """
        if self._pos_table is None:
            return ()
        positions = self._pos_table.value(exclude_unchecked, exclude_hidden_cols)
        if isinstance(positions, useq.WellPlatePlan):
            return positions
        return tuple(positions)

    def assign_wells_to_positions(self) -> None:
        """Fill in the well of each position in the bound table.

        Each position is assigned the well whose centroid is closest to it, and
        the well id lands in the table's "Well" column, which is stored on the
        position as `plate_row`/`plate_col`.  Those are the fields the OME-Zarr
        writers read to build the plate/well hierarchy of the dataset, so the
        well ends up in the data as real metadata rather than as part of a name.

        Position names are left untouched, and re-running the assignment simply
        overwrites the wells.
        """
        plan = self._plan
        if plan is None or not self._calibrated:
            QMessageBox.information(
                self,
                "No calibrated plate",
                "Select and calibrate a well plate before assigning wells.",
            )
            return

        if self._pos_table is None:
            QMessageBox.information(
                self,
                "No positions",
                "The map is not showing any position list.\nPick one from the "
                "positions button in the toolbar.",
            )
            return

        # a table driven by the HCS wizard regenerates its own positions from
        # its plate plan, so edits there would not stick
        if getattr(self._pos_table, "_plate_plan", None) is not None:
            QMessageBox.information(
                self,
                "Positions are generated from a plate plan",
                "These positions come from the HCS wizard, which assigns them "
                "to wells from its own plate plan.\nMake them editable first, "
                "or point the map at another position table.",
            )
            return

        table = self._pos_table.table()
        well_key = self._pos_table.WELL.key
        with signals_blocked(self._pos_table):
            for row in range(table.rowCount()):
                data = table.rowData(row)
                x, y = data.get(self._pos_table.X.key), data.get(self._pos_table.Y.key)
                if x is None or y is None:  # pragma: no cover
                    continue
                hit = nearest_well(plan, float(x), float(y))
                table.setRowData(row, {well_key: hit.name})
        self._pos_table.valueChanged.emit()
        self.wellsAssigned.emit()

    # -----------------------------PRIVATE METHODS------------------------------------

    # PLATE / CALIBRATION ----------------------------------------------------

    def _remember_plate(self, plate: useq.WellPlate) -> None:
        """Keep track of plates that are not in the useq registry."""
        name = plate.name or f"{plate.rows}x{plate.columns} plate"
        if not plate.name:
            plate = plate.model_copy(update={"name": name})
        self._custom_plates[name] = plate

    def _plate_for_name(self, name: str) -> useq.WellPlate | None:
        if name in self._custom_plates:
            return self._custom_plates[name]
        try:
            return useq.WellPlate.from_str(name)
        except (KeyError, ValueError):
            return None

    def _select_combo_name(self, name: str) -> None:
        """Set the plate combo to `name` (adding it if needed), without side effects."""
        combo = self._toolbar.plate_combo
        if combo.findText(name) < 0:
            combo.addItem(name)
        with signals_blocked(combo):
            combo.setCurrentText(name)

    def _on_plate_combo_changed(self, name: str) -> None:
        if (plate := self._plate_for_name(name)) is None:
            self._set_plan(None, calibrated=False)
        else:
            self._apply_plate(plate)

    def _apply_plate(self, plate: useq.WellPlate) -> None:
        """Set `plate` as the current plate, loading a stored calibration if any."""
        if (stored := self._store.load(plate.name)) is not None:
            self._set_plan(stored, calibrated=True)
        else:
            # preview the (uncalibrated) plate with A1 at the stage origin
            plan = useq.WellPlatePlan(plate=plate, a1_center_xy=(0.0, 0.0))
            self._set_plan(plan, calibrated=False)

    def _set_plan(self, plan: useq.WellPlatePlan | None, *, calibrated: bool) -> None:
        self._plan = plan
        self._calibrated = calibrated and plan is not None
        self._last_px_per_um = None  # new plate: labels need resizing
        self._update_plate_transform()
        self._plate_overlay.set_plan(plan, calibrated=self._calibrated)
        self._update_status()
        self._update_action_enablement()
        self.platePlanChanged.emit(self.platePlan())
        if plan is not None:
            self.zoom_to_fit()

    def calibrate(self) -> _PlateCalibrationDialog | None:
        """Open the (non-modal) plate calibration dialog, returning it.

        Calibrating means driving the stage to the center of at least three
        wells, so the dialog is deliberately *not* modal: the map, the stage
        controls and everything else stay usable while it is open.  On accept,
        the calibration is stored and applied.
        """
        if self._plan is None:
            return None
        if self._calib_dialog is not None:
            self._calib_dialog.raise_()
            return self._calib_dialog

        self._calib_dialog = dialog = _PlateCalibrationDialog(
            self._plan if self._calibrated else self._plan.plate,
            parent=self,
            mmcore=self._mmc,
        )
        dialog.accepted.connect(self._on_calibration_accepted)
        dialog.finished.connect(self._on_calibration_dialog_finished)
        dialog.show()
        return dialog

    def _on_calibration_accepted(self) -> None:
        if self._calib_dialog is not None and (
            (plan := self._calib_dialog.value()) is not None
        ):
            self._store.save(plan)
            self._set_plan(plan, calibrated=True)

    def _on_calibration_dialog_finished(self) -> None:
        if (dialog := self._calib_dialog) is not None:
            self._calib_dialog = None
            dialog.deleteLater()

    def _delete_calibration(self) -> None:
        if self._plan is None:  # pragma: no cover
            return
        plate = self._plan.plate
        if (
            QMessageBox.question(
                self,
                "Delete Calibration",
                f"Delete the stored calibration for {plate.name!r}?\n"
                "The plate overlay will revert to an uncalibrated preview.",
            )
            != QMessageBox.StandardButton.Yes
        ):
            return
        self._store.delete(plate.name)
        plan = useq.WellPlatePlan(plate=plate, a1_center_xy=(0.0, 0.0))
        self._set_plan(plan, calibrated=False)

    def _export_calibration(self) -> None:
        if not self._calibrated or self._plan is None:  # pragma: no cover
            return
        file, _ = QFileDialog.getSaveFileName(
            self, "Export Plate Calibration", "", "json(*.json)"
        )
        if file:
            dest = Path(file)
            if not dest.suffix:
                dest = dest.with_suffix(".json")
            dest.write_text(self._plan.model_dump_json(exclude_unset=True, indent=2))

    def _import_calibration(self) -> None:
        file, _ = QFileDialog.getOpenFileName(
            self, "Import Plate Calibration", "", "json(*.json)"
        )
        if not file:
            return
        try:
            plan = useq.WellPlatePlan.model_validate_json(Path(file).read_text())
        except ValueError as e:
            QMessageBox.warning(self, "Import Error", f"Invalid calibration file:\n{e}")
            return
        self._store.save(plan)
        self.setPlatePlan(plan)

    def _update_status(self) -> None:
        if self._plan is None:
            txt = "Select a plate format to overlay it on the map."
            tooltip = ""
        elif self._calibrated:
            x, y = self._plan.a1_center_xy
            rot = self._plan.rotation or 0
            name = self._plan.plate.name or "plate"
            txt = (
                f"<font color='{GREEN}'>●</font> <b>{name}</b> calibrated: "
                f"A1 at ({x / 1000:.2f}, {y / 1000:.2f}) mm, rotation {rot:.2f}°"
            )
            tooltip = f"Calibration stored in:\n{self._store.path_for(name)}"
        else:
            name = self._plan.plate.name or "plate"
            txt = (
                f"<font color='{ORANGE}'>●</font> <b>{name}</b> not calibrated. "
                "overlay is a preview with A1 at (0, 0). "
                "Use the calibrate button to align it with the stage."
            )
            tooltip = ""
        self._status_label.setText(txt)
        self._status_label.setToolTip(tooltip)

    def _update_action_enablement(self) -> None:
        tb = self._toolbar
        has_plate = self._plan is not None
        tb.calibrate_action.setEnabled(has_plate)
        tb.assign_action.setEnabled(self._calibrated and self._pos_table is not None)
        tb.export_action.setEnabled(self._calibrated)
        tb.delete_action.setEnabled(
            has_plate
            and self._plan is not None
            and self._store.has_calibration(self._plan.plate.name)
        )

    # APPEARANCE --------------------------------------------------------------

    def _apply_palette(self) -> None:
        """Take the canvas colors from the widget palette.

        This keeps the map in step with whatever application hosts it (napari's
        dark theme, a light desktop theme, ...) without knowing anything about
        that application.
        """
        palette = self.palette()
        bg = palette.color(QPalette.ColorRole.Window)
        fg = palette.color(QPalette.ColorRole.WindowText)
        mid = palette.color(QPalette.ColorRole.Mid)

        # selection must depart from POSITION_COLOR toward the background's
        # opposite: brighter blue on a dark theme, darker on a light one
        self._positions_overlay.set_selected_color(
            SELECTED_COLOR_DARK_BG if bg.lightness() < 128 else SELECTED_COLOR_LIGHT_BG
        )

        self._stage_viewer.canvas.bgcolor = _rgba(bg)
        self._map_frame.setStyleSheet(
            f"QFrame#stageMapFrame {{ border: 1px solid {mid.name()}; "
            f"background: {bg.name()}; }}"
        )
        # the plate is a backdrop for the positions, so it is drawn muted
        self._plate_overlay.set_colors(_rgba(fg, 0.5), _rgba(fg, 0.22))

    def _update_label_font_sizes(self) -> None:
        """Scale the well ids so they stay proportional to the wells.

        Clamped to a readable range, and dropped entirely once a well is too
        small for its id to be legible.  Position names are left at a fixed
        size: unlike the well ids they annotate something that can be far
        smaller than a pixel when zoomed out.

        Runs on every canvas draw, so it must be a no-op at constant zoom:
        vispy schedules a repaint on every visual touch (even a same-value
        ``visible`` assignment), which would otherwise sustain a redraw loop.
        """
        canvas_width = self._stage_viewer.canvas.size[0]
        rect_width = self._stage_viewer.view.camera.rect.width
        if not canvas_width or not rect_width:  # pragma: no cover
            return
        px_per_um = canvas_width / rect_width
        if px_per_um == self._last_px_per_um:
            return
        self._last_px_per_um = px_per_um
        self._update_label_offsets()

        if self._plan is not None:
            well_px = self._plan.plate.well_size[0] * 1000 * px_per_um
            self._plate_overlay.set_label_font_size(min(38.0, max(6.0, well_px * 0.24)))
            self._plate_overlay.set_labels_visible(
                self._toolbar.labels_action.isChecked() and well_px > 26
            )

    # POSITIONS ---------------------------------------------------------------

    def _fov_size(self) -> tuple[float, float] | None:
        """Return the camera field of view in µm, or None if it is unknown.

        `getImageWidth`/`getImageHeight` report the *current* image size, so any
        ROI crop and binning are already accounted for.
        """
        px = self._mmc.getPixelSizeUm()
        w, h = self._mmc.getImageWidth(), self._mmc.getImageHeight()
        if px <= 0 or not w or not h:
            return None
        return w * px, h * px

    def _refresh_positions(self) -> None:
        positions = (
            list(self._pos_table.value(exclude_unchecked=False))
            if self._pos_table is not None
            else []
        )
        xy = [(pos.x or 0.0, pos.y or 0.0) for pos in positions]
        # name above the marker, inferred well id below it
        names = [pos.name or "" for pos in positions]
        wells = [
            well_id(pos.plate_row, pos.plate_col)
            if pos.plate_row is not None and pos.plate_col is not None
            else ""
            for pos in positions
        ]
        self._last_fov_size = self._fov_size()
        self._pos_xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        self._positions_overlay.set_positions(xy, names, wells, self._last_fov_size)
        self._update_label_offsets()
        self._sync_selection_from_table()

    def _update_label_offsets(self) -> None:
        """Keep the labels a constant screen distance from their marker.

        The clearance is a scene-space vector (labels live in stage µm), so
        zoom and plate-transform changes both require a recompute.
        """
        ov = self._positions_overlay
        if self._pos_xy is None or not len(self._pos_xy):
            return
        try:
            tr = ov.label_to_canvas_transform()
            origin = tr.map((0.0, 0.0))[:2]
            # the scene point 24 px above the scene origin on screen. vispy
            # Text anchors align to the glyph bbox ascender, which absorbs
            # ~10 px, so the visible gap is roughly offset - 10 px - marker
            # radius: 24 px leaves the text almost touching the dot
            up = tr.imap((origin[0], origin[1] - 24.0))[:2]
        except Exception:
            return  # transforms not ready yet (first draw pending)
        ov.set_label_offset(np.asarray(up, dtype=float))

    def _sync_selection_from_table(self) -> None:
        """Mirror the bound table's selected rows as highlighted markers."""
        if self._pos_table is None:
            self._positions_overlay.set_selected(())
            return
        rows = {i.row() for i in self._pos_table.table().selectedIndexes()}
        self._positions_overlay.set_selected(rows)

    def _position_at(self, canvas_pos: Sequence[float]) -> int | None:
        """Return the index of the position under `canvas_pos`, or None."""
        if self._pos_xy is None or not len(self._pos_xy):
            return None
        canvas_width = self._stage_viewer.canvas.size[0]
        rect_width = self._stage_viewer.view.camera.rect.width
        if not canvas_width or not rect_width:  # pragma: no cover
            return None
        # click position in plate space, positions mapped stage -> plate
        plate_click = self._stage_viewer.view.camera.transform.imap(canvas_pos)[:2]
        plate_pts = self._plate_node.transform.map(self._pos_xy)[:, :2]
        d2 = np.sum((plate_pts - plate_click) ** 2, axis=1)
        idx = int(np.argmin(d2))
        # hit radius: a comfortable ~14 px around the marker, in µm
        radius_um = 14 * rect_width / canvas_width
        return idx if d2[idx] <= radius_um**2 else None

    def _update_fov_size(self) -> None:
        """Resize everything that depends on the camera field of view.

        Called from core events (pixelSizeChanged, roiSet, binning /
        camera-device propertyChanged, systemConfigurationLoaded) and from the
        background poller's fovChanged, which backstops changes that emit no
        event (clearROI, a python camera resizing its own ROI). Never call
        this from a GUI-thread timer: core accessors can block on a device
        module lock held by the poller during a slow serial read.
        """
        if (fov := self._fov_size()) is None or fov == self._last_fov_size:
            return
        self._affine_state.refresh()
        w, h = self._mmc.getImageWidth(), self._mmc.getImageHeight()
        # the marker rect is in camera pixels: its transform applies pixel scaling
        self._stage_pos_marker.set_rect_size(w, h)
        self._refresh_positions()

    def _populate_source_menu(self) -> None:
        """(Re)build the menu listing the position tables that can be shown."""
        menu = self._toolbar.source_menu
        menu.clear()

        tables = self.linkablePositionTables()
        if not tables:
            no_table = menu.addAction("No position table found")
            no_table.setEnabled(False)
        for title, table in tables.items():
            action = menu.addAction(title)
            action.setCheckable(True)
            action.setChecked(table is self._pos_table)
            action.triggered.connect(
                lambda _checked=False, t=table: self.setPositionTable(t)
            )

        menu.addSeparator()
        none_action = menu.addAction("Show no positions")
        none_action.setCheckable(True)
        none_action.setChecked(self._pos_table is None)
        none_action.triggered.connect(lambda: self.setPositionTable(None))

    def _update_source_label(self) -> None:
        self._toolbar.set_action_active(
            self._toolbar.positions_action,
            "mdi:format-list-bulleted",
            self._pos_table is not None,
        )
        if self._pos_table is None:
            self._source_label.setText("<i>no position list</i>")
            self._source_label.setToolTip(
                "Pick a position list from the positions button in the toolbar"
            )
        else:
            title = _table_title(self._pos_table)
            self._source_label.setText(f"<i>positions: <b>{title}</b></i>")
            self._source_label.setToolTip("")

    def _on_table_destroyed(self) -> None:
        """Unbind when the table we were showing goes away.

        The C++ object is already gone here, so its signals must not be touched
        (which rules out going through setPositionTable).
        """
        self._pos_table = None
        self._update_source_label()
        self._update_action_enablement()
        self._refresh_positions()
        self.positionTableChanged.emit(None)

    def _on_trails_toggled(self, checked: bool) -> None:
        self._positions_overlay.set_trail_visible(checked)

    def _on_labels_toggled(self, checked: bool) -> None:
        self._plate_overlay.set_labels_visible(checked)
        self._positions_overlay.set_labels_visible(checked)
        self._last_px_per_um = None  # re-evaluate label size + visibility

    # STAGE -------------------------------------------------------------------

    def _update_stage_controller(self) -> None:
        # cache the label so event handlers never ask the core for it
        self._xy_device = xy_device = self._mmc.getXYStageDevice()
        if xy_device:
            self._stage_controller = QStageMoveAccumulator.for_device(
                xy_device, self._mmc
            )
        else:
            self._stage_controller = None
            self._stage_pos_label.setText("No XY stage device")

    def _on_mouse_release(self, event: MouseEvent) -> None:
        """Single left click on a position selects it in the bound table.

        A drag (pan) is not a click: the release must land within a few px of
        the press. Ctrl-click toggles, so several positions can be gathered;
        clicking empty map clears the selection. The selection highlight comes
        back via the table's itemSelectionChanged -> _sync_selection_from_table.
        """
        if self._pos_table is None or event.button != 1:
            return
        press = getattr(event, "press_event", None)
        if press is None:
            return
        dx, dy = event.pos[0] - press.pos[0], event.pos[1] - press.pos[1]
        if dx * dx + dy * dy > 25:  # moved > 5 px: a pan, not a click
            return
        tbl = self._pos_table.table()
        idx = self._position_at(event.pos)
        if idx is None:
            tbl.clearSelection()
            return
        ctrl = any(
            getattr(k, "name", "") == "Control" for k in (event.modifiers or ())
        )
        flags = QItemSelectionModel.SelectionFlag.Rows | (
            QItemSelectionModel.SelectionFlag.Toggle
            if ctrl
            else QItemSelectionModel.SelectionFlag.ClearAndSelect
        )
        model_index = tbl.model().index(idx, 0)
        tbl.selectionModel().select(model_index, flags)
        tbl.scrollTo(model_index)

    def _on_mouse_double_click(self, event: MouseEvent) -> None:
        """Move the stage to the double-clicked position."""
        if self._stage_controller is None:
            return
        # camera.imap gives canonical plate coords (view.scene space); undo the
        # plate transform to get back the STAGE coordinate to move to.
        plate = self._stage_viewer.view.camera.transform.imap(event.pos)
        x, y, *_ = self._plate_node.transform.imap(plate)
        self._stage_controller.move_absolute((x, y))
        self._stage_pos_label.setText(f"X: {x:.2f} µm  Y: {y:.2f} µm")

    def _on_poll_stage_toggled(self, checked: bool) -> None:
        self._polling = checked
        self._marker_visible = checked
        self._stage_pos_marker.visible = checked
        if checked:
            self._stage_poller.start()  # background position reads (off GUI thread)
        else:
            self._stage_poller.stop()

    def _on_xy_event(self, device: str, stage_x: float, stage_y: float) -> None:
        """Apply a position reported by the core callback (commanded moves).

        Instant and free; the poller covers only what the adapter never
        reports (joystick moves on serial stages).
        """
        if device == getattr(self, "_xy_device", ""):
            self._on_stage_position_polled(stage_x, stage_y)

    def _on_stage_position_polled(self, stage_x: float, stage_y: float) -> None:
        """Update the marker + label from a background-polled stage position.

        Runs on the GUI thread (queued from the poller thread) and only fires
        when the stage actually moved, so it neither blocks nor churns the map.
        """
        if self._closing:  # a signal may already be queued when close begins
            return
        self._last_stage_pos = (stage_x, stage_y)
        txt = f"X: {stage_x:.2f} µm  Y: {stage_y:.2f} µm"
        if self._calibrated and self._plan is not None:
            hit = nearest_well(self._plan, stage_x, stage_y)
            if hit.inside:
                txt += f"   [{hit.name}]"
        self._stage_pos_label.setText(txt)
        # skip the GL marker draw across a reparent (context torn down); the
        # position is recorded above and re-applied by _on_gl_reset() on show.
        if self._gl_suspended:
            return
        # fast path: copy cached rotation/scale part and just update translation
        matrix = self._affine_state.system_affine_translated(stage_x, stage_y)
        self._stage_pos_marker.apply_transform(matrix.T)

    def _on_canvas_draw(self, event: object = None) -> None:
        """Rescale label fonts whenever the map redraws (zoom/pan/resize).

        Never make core calls here: even cheap accessors like
        ``getPixelSizeUm`` can block on a device module lock that the poller
        thread holds during a slow serial read. Field-of-view changes arrive
        via core events (``pixelSizeChanged``, ``roiSet``,
        ``propertyChanged``, ``systemConfigurationLoaded``) and the poller
        thread's ``fovChanged``.

        A font change schedules one follow-up draw; the px_per_um cache in
        ``_update_label_font_sizes`` makes that pass a no-op, so this cannot
        self-sustain a redraw loop.
        """
        if self._gl_suspended or self._closing:  # reparent / teardown in flight
            return
        self._update_label_font_sizes()
        # Recompute the label clearance on every draw: it is pure math, a
        # no-op when unchanged, and a draw is the only moment the transform
        # chain is guaranteed live (outside one, e.g. right after a GL reset,
        # the computation can silently fail).
        self._update_label_offsets()

    def _on_canvas_resize(self, event: object = None) -> None:
        """Keep the map fitted to the canvas while the view is untouched.

        The camera keeps its world rect across a canvas resize, so growing the
        widget just adds empty margin. As long as the camera still shows
        exactly the last ``zoom_to_fit`` rect (the user has not zoomed or
        panned since), re-fit to the new canvas. A hand-adjusted view is never
        overridden.
        """
        if self._gl_suspended or self._closing or self._fit_rect is None:
            return
        r = self._stage_viewer.view.camera.rect
        cur = (r.left, r.bottom, r.width, r.height)
        scale = max(abs(v) for v in self._fit_rect) or 1.0
        if all(abs(c - f) <= scale * 1e-6 for c, f in zip(cur, self._fit_rect)):
            self.zoom_to_fit()

    def _build_stage_marker(self) -> None:
        """Create the stage-position marker on the scene (fresh GL program)."""
        w = self._mmc.getImageWidth() or 512
        h = self._mmc.getImageHeight() or 512
        self._stage_pos_marker = StagePositionMarker(
            parent=self._plate_node,
            rect_width=w,
            rect_height=h,
            # screen px: the crosshair keeps a constant size at any zoom
            marker_symbol_size=16,
        )
        self._stage_pos_marker.visible = self._marker_visible

    # CORE EVENTS -------------------------------------------------------------

    def _on_sys_config_loaded(self) -> None:
        self._affine_state.refresh()
        w = self._mmc.getImageWidth() or 512
        h = self._mmc.getImageHeight() or 512
        self._stage_pos_marker.set_rect_size(w, h)
        self._update_stage_controller()
        self._last_fov_size = None
        self._refresh_positions()

    def _update_plate_transform(self) -> None:
        """Set the stage->plate transform so the plate draws canonically.

        Everything under ``_plate_node`` is drawn in STAGE-µm coordinates; this
        maps them to canonical plate space, exactly the local frame ``nearest_well``
        uses: ``local = R(-rotation) @ (stage - a1_center)`` (columns -> +x, rows
        -> -y), so A1 is always top-left regardless of how the stage axes are
        wired. The marker (drawn at the raw stage position) is mapped by the same
        transform, so it shows the TRUE position on the fixed plate: if the
        calibration/axes are wrong the marker is visibly off, rather than the
        plate silently rotating. Identity when there is no plan (plain stage
        space), which is also what an uncalibrated preview (a1=0, rot=0) yields.
        """
        m = np.eye(4)
        if (plan := self._plan) is not None:
            theta = np.deg2rad(plan.rotation or 0.0)
            cos_, sin_ = np.cos(-theta), np.sin(-theta)
            rot = np.array([[cos_, -sin_], [sin_, cos_]])  # R(-rotation)
            a1 = np.asarray(plan.a1_center_xy, dtype=float)
            m[:2, :2] = rot
            m[:2, 3] = -rot @ a1
        # vispy MatrixTransform uses row-vector convention (out = in @ M), so
        # pass the transpose of our column-convention matrix (as apply_transform).
        self._plate_node.transform = MatrixTransform(matrix=m.T)
        # the label clearance vector lives in stage µm and depends on this
        # transform's direction (a rotated/reflected calibration flips it);
        # the zoom cache alone would not notice the change
        self._update_label_offsets()

    def _on_pixel_size_changed(self, value: float = 0.0) -> None:
        self._affine_state.refresh()
        self._update_fov_size()

    def _on_roi_changed(self) -> None:
        self._update_fov_size()

    def _on_property_changed(self, device: str, prop: str, value: str = "") -> None:
        """Catch field-of-view changes that emit no dedicated event.

        Binning changes only emit propertyChanged, and swapping the Core
        camera / XY stage device likewise.
        """
        if device == "Core":
            if prop == "Camera":
                self._update_fov_size()
            elif prop == "XYStage":
                self._update_stage_controller()
        elif prop == "Binning":
            self._update_fov_size()

    def _on_fov_polled(self, px_size: float, width: int, height: int) -> None:
        """Apply a field-of-view change spotted by the background poller.

        The poller only emits when the FOV actually differs, so this runs
        rarely; it backstops changes with no core event at all (``clearROI``,
        a python camera resizing its own ROI).
        """
        if self._closing:
            return
        self._update_fov_size()

    def _disconnect(self) -> None:
        # Idempotent: both closeEvent and the `destroyed` signal may call this,
        # so whichever runs first tears down and the other is a no-op.
        if getattr(self, "_disconnected", False):
            return
        self._disconnected = True
        # Latch teardown first so any already-queued deferred redraw / poller
        # signal / timer tick becomes a no-op instead of touching a dying
        # canvas or core.
        self._closing = True
        # Stop the background thread before its C++ object is destroyed with the
        # widget; a QThread still running at destruction aborts the process.
        with suppress(Exception):
            self._stage_poller.stop()
        events = self._mmc.events
        for sig, slot in (
            (events.systemConfigurationLoaded, self._on_sys_config_loaded),
            (events.pixelSizeChanged, self._on_pixel_size_changed),
            (events.roiSet, self._on_roi_changed),
            (events.propertyChanged, self._on_property_changed),
            (events.XYStagePositionChanged, self._on_xy_event),
        ):
            with suppress(Exception):
                sig.disconnect(slot)


class _PlateCalibrationDialog(QDialog):
    """Dialog wrapping a PlateCalibrationWidget with Ok/Cancel buttons."""

    def __init__(
        self,
        plate_or_plan: useq.WellPlate | useq.WellPlatePlan,
        parent: QWidget | None = None,
        mmcore: CMMCorePlus | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Calibrate Well Plate")

        from pymmcore_widgets.hcs._plate_calibration_widget import (
            PlateCalibrationWidget,
        )

        self._calibration = PlateCalibrationWidget(self, mmcore)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        self._ok_btn = btns.button(QDialogButtonBox.StandardButton.Ok)

        layout = QVBoxLayout(self)
        layout.addWidget(self._calibration, 1)
        layout.addWidget(btns, 0)

        self._calibration.calibrationChanged.connect(self._on_calibration_changed)
        self._calibration.setValue(plate_or_plan)

    def value(self) -> useq.WellPlatePlan | None:
        """Return the calibrated plate plan."""
        return self._calibration.value()

    def _on_calibration_changed(self, calibrated: bool) -> None:
        if self._ok_btn is not None:
            self._ok_btn.setEnabled(calibrated)


class _StageMapToolbar(QToolBar):
    """Toolbar for the StageMapWidget.

    Follows the icon conventions used by the other widgets in this package: flat
    iconify glyphs with no button chrome, gray when off and green when on, so
    the toolbar blends into whichever application hosts the widget.
    """

    if TYPE_CHECKING:

        def addAction(self, icon: QIcon, text: str) -> QAction: ...

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setIconSize(QSize(22, 22))
        self.setContentsMargins(0, 0, 8, 0)

        # plate format selector
        self.plate_combo = QComboBox()
        self.plate_combo.setToolTip("Well plate format to overlay on the map")
        plate_names = sorted(useq.registered_well_plate_keys(), key=_sort_plate)
        self.plate_combo.addItems([NO_PLATE, *plate_names])
        lbl = QLabel("Plate:")
        lbl.setContentsMargins(6, 0, 4, 0)
        self.addWidget(lbl)
        self.addWidget(self.plate_combo)

        # calibration (with a menu for import/export/delete)
        self.calibrate_action = self._add_action(
            "mdi:crosshairs-gps", "Calibrate Plate..."
        )
        self.import_action = QAction(
            QIconifyIcon("mdi:file-import-outline", color=GRAY),
            "Import Calibration...",
            self,
        )
        self.export_action = QAction(
            QIconifyIcon("mdi:file-export-outline", color=GRAY),
            "Export Calibration...",
            self,
        )
        self.delete_action = QAction(
            QIconifyIcon("mdi:delete-outline", color=RED),
            "Delete Stored Calibration",
            self,
        )
        calib_menu = QMenu(self)
        calib_menu.addAction(self.import_action)
        calib_menu.addAction(self.export_action)
        calib_menu.addSeparator()
        calib_menu.addAction(self.delete_action)
        self._attach_menu(self.calibrate_action, calib_menu)

        self.assign_action = self._add_action(
            "mdi:tag-multiple-outline", "Assign Positions to Nearest Well"
        )

        self.addSeparator()

        self.trails_action = self._add_action(
            "mdi:vector-polyline", "Show Travel Path", checkable=True, checked=True
        )
        self.labels_action = self._add_action(
            "mdi:format-letter-case", "Show Labels", checkable=True, checked=True
        )
        # picks which position list the map reflects (this widget has none)
        self.positions_action = self._add_action(
            "mdi:format-list-bulleted", "Position List to Show"
        )
        self.source_menu = QMenu(self)
        self._attach_menu(
            self.positions_action,
            self.source_menu,
            QToolButton.ToolButtonPopupMode.InstantPopup,
        )

        self.addSeparator()

        self.poll_action = self._add_action(
            "mdi:map-marker-outline", "Show Stage Position", checkable=True
        )
        self.grid_action = self._add_action("mdi:grid", "Show Grid", checkable=True)
        self.zoom_to_fit_action = self._add_action("mdi:fullscreen", "Zoom to Fit")

    def _add_action(
        self,
        glyph: str,
        text: str,
        *,
        checkable: bool = False,
        checked: bool = False,
    ) -> QAction:
        """Add a toolbar action, recoloring its icon when toggled."""
        action = self.addAction(QIconifyIcon(glyph, color=GRAY), text)
        action.setToolTip(text)
        if checkable:
            action.setCheckable(True)

            def _recolor(
                on: bool, glyph: str = glyph, action: QAction = action
            ) -> None:
                action.setIcon(QIconifyIcon(glyph, color=GREEN if on else GRAY))

            action.toggled.connect(_recolor)
            action.setChecked(checked)
            _recolor(checked)
        return action

    def set_action_active(self, action: QAction, glyph: str, active: bool) -> None:
        """Color a non-checkable action as if it were on."""
        action.setIcon(QIconifyIcon(glyph, color=GREEN if active else GRAY))

    def _attach_menu(
        self,
        action: QAction,
        menu: QMenu,
        mode: QToolButton.ToolButtonPopupMode = (
            QToolButton.ToolButtonPopupMode.MenuButtonPopup
        ),
    ) -> None:
        """Give `action`'s tool button a drop-down menu."""
        btn = cast("QToolButton", self.widgetForAction(action))
        btn.setMenu(menu)
        btn.setPopupMode(mode)
