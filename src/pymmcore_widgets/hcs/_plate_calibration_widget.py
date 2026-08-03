from __future__ import annotations

from contextlib import suppress
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
import useq
from pymmcore_plus import CMMCorePlus
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedWidget,
    QStyle,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from superqt.iconify import QIconifyIcon

from pymmcore_widgets._util import SeparatorWidget
from pymmcore_widgets.hcs._well_calibration_widget import (
    CALIBRATED_ICON,
    GREEN,
    WellCalibrationWidget,
)
from pymmcore_widgets.useq_widgets._well_plate_widget import WellPlateView

if TYPE_CHECKING:
    from collections.abc import Mapping


class CalibrationMode(str, Enum):
    """How many wells are used to calibrate a plate, and what is solved for."""

    TWO_POINT = "2 wells (no rotation)"
    THREE_POINT = "3 wells (with rotation)"

    def __str__(self) -> str:
        return self.value

    @property
    def wells_required(self) -> int:
        """Number of calibrated wells needed by this mode."""
        return 2 if self is CalibrationMode.TWO_POINT else 3

    @property
    def solves_rotation(self) -> bool:
        """Whether the plate rotation is fitted (rather than assumed to be zero)."""
        return self is CalibrationMode.THREE_POINT


class PlateCalibrationWidget(QWidget):
    """Widget to calibrate a well plate.

    Provides a view of the well plate with the ability to select and calibrate
    individual wells.

    Two calibration modes are offered: two wells, which fits the plate origin
    and spacing but assumes the plate is square with the stage axes, and three
    wells, which additionally fits the rotation.  Two wells is quicker and is
    enough for a plate held in a fixed, well machined holder; use three
    whenever the plate can sit at a slight angle.
    """

    calibrationChanged = Signal(bool)

    def __init__(
        self, parent: QWidget | None = None, mmcore: CMMCorePlus | None = None
    ) -> None:
        super().__init__(parent)

        self._mmc = mmcore or CMMCorePlus.instance()
        self._current_plate: useq.WellPlate | None = None
        self._a1_center_xy: tuple[float, float] = (0.0, 0.0)
        self._well_spacing: tuple[float, float] | None = None
        self._rotation: float | None = None
        self._mode: CalibrationMode = CalibrationMode.THREE_POINT

        # mapping of well index (r, c) to well center (x, y)
        self._calibrated_wells: dict[tuple[int, int], tuple[float, float]] = {}

        # WIDGETS ------------------------------------------------------------

        self._mode_combo = QComboBox()
        self._mode_combo.addItems([str(m) for m in CalibrationMode])
        self._mode_combo.setCurrentText(str(self._mode))
        self._mode_combo.setToolTip(
            "Two wells fit the plate origin and spacing, assuming the plate is "
            "square with the stage axes.\nThree wells also fit the rotation."
        )

        self._tab_wdg = QTabWidget()

        self._plate_view = WellPlateView()
        self._plate_view.setDragMode(WellPlateView.DragMode.NoDrag)
        self._plate_view.setSelectionMode(WellPlateView.SelectionMode.SingleSelection)
        self._plate_view.setSelectedColor(Qt.GlobalColor.yellow)

        self._plate_test = WellPlateView()
        lbl = QLabel("double-click on point to move stage", self._plate_test)
        lbl.setGeometry(4, 0, 200, 20)
        lbl.setStyleSheet("color: #CCC; font-size: 10px; background: transparent;")
        self._plate_test.setDragMode(WellPlateView.DragMode.NoDrag)
        self._plate_test.setSelectionMode(WellPlateView.SelectionMode.NoSelection)
        self._plate_test.setDrawWellEdgeSpots(True)
        self._plate_test.setDrawLabels(False)

        self._tab_wdg.addTab(self._plate_view, "Calibrate Plate")
        self._tab_wdg.addTab(self._plate_test, "Test Calibration")
        self._tab_wdg.setTabEnabled(1, False)

        self._test_well_btn = QPushButton("Test Well", self)
        self._test_well_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._test_well_btn.setEnabled(False)

        # mapping of well index (r, c) to calibration widget
        # these are created on demand in _get_or_create_well_calibration_widget
        self._calibration_widgets: dict[tuple[int, int], WellCalibrationWidget] = {}
        self._calibration_widget_stack = QStackedWidget()

        self._info = QLabel("Please calibrate the plate.")
        self._info_icon = QLabel()
        self._update_info()

        # LAYOUT -------------------------------------------------------------

        mode_row = QHBoxLayout()
        mode_row.setContentsMargins(0, 0, 0, 0)
        mode_row.addWidget(QLabel("Calibrate with:"), 0)
        mode_row.addWidget(self._mode_combo, 1)

        right_wdg = QGroupBox()
        right_layout = QVBoxLayout(right_wdg)
        right_layout.setContentsMargins(5, 5, 5, 5)
        right_layout.addLayout(mode_row)
        right_layout.addWidget(SeparatorWidget())
        right_layout.addWidget(self._calibration_widget_stack)
        right_layout.addWidget(SeparatorWidget())
        right_layout.addWidget(self._test_well_btn)
        right_layout.addStretch()

        top = QHBoxLayout()
        top.addWidget(right_wdg, 0)
        top.addWidget(self._tab_wdg, 1)

        info_layout = QHBoxLayout()
        info_layout.addWidget(self._info_icon, 0)
        info_layout.addWidget(self._info, 1)

        main = QVBoxLayout(self)
        main.addLayout(top)
        main.addLayout(info_layout)

        # CONNECTIONS ---------------------------------------------------------

        self._mode_combo.currentTextChanged.connect(self._on_mode_changed)
        self._plate_view.selectionChanged.connect(self._on_plate_selection_changed)
        self._tab_wdg.currentChanged.connect(self._on_tab_changed)
        self._plate_test.positionDoubleClicked.connect(self._move_to_xy_position)
        self._test_well_btn.clicked.connect(self._move_to_test_position)

    # ---------------------------PUBLIC API-----------------------------------

    def setValue(
        self, plate_or_plan: str | useq.WellPlate | useq.WellPlatePlan
    ) -> None:
        """Set the plate to be calibrated.

        Parameters
        ----------
        plate_or_plan : str | useq.WellPlate | useq.WellPlatePlan
            The well plate to calibrate. If a string, it is assumed to be the name of a
            well plate (e.g. "96-well"). If a WellPlate instance, it is used directly.
            If a WellPlatePlan instance, the plate is set to the plan's plate and the
            widget state is set to calibrated.
        """
        calibrated: bool = False
        plan: useq.WellPlatePlan | None = None
        plate: useq.WellPlate | None = None
        if isinstance(plate_or_plan, str):
            plate = useq.WellPlate.from_str(plate_or_plan)
        elif isinstance(plate_or_plan, useq.WellPlatePlan):
            self._a1_center_xy = plate_or_plan.a1_center_xy
            self._rotation = plate_or_plan.rotation
            plan = plate_or_plan
            plate = plan.plate
            calibrated = True
            # set the well spacing to the plate's well spacing to correctly update the
            # info label (since we are assuming the plate is correctly calibrated,
            # we can assume the well spacing is correct)
            self._well_spacing = plate.well_spacing
        elif isinstance(plate_or_plan, useq.WellPlate):
            plate = plate_or_plan

        self._current_plate = plate
        self._plate_view.drawPlate(plate)

        # clear existing calibration widgets
        while self._calibration_widgets:
            wdg = self._calibration_widgets.popitem()[1]
            self._calibration_widget_stack.removeWidget(wdg)
            wdg.deleteLater()

        self._plate_view.setSelectedIndices([(0, 0)])  # select A1

        if calibrated and plan is not None:
            self._plate_test.drawPlate(plan)
            # make sure no wells are selected since we are in test mode
            self._plate_test.setSelectedIndices([])
            # set the tab to the test mode
            self._tab_wdg.setCurrentIndex(1)
        else:
            self._plate_test.clear()

        self._update_info()
        self._tab_wdg.setTabEnabled(1, calibrated)
        self.calibrationChanged.emit(calibrated)

    def calibrationMode(self) -> CalibrationMode:
        """Return the current calibration mode."""
        return self._mode

    def setCalibrationMode(self, mode: CalibrationMode | str) -> None:
        """Set whether two wells (no rotation) or three wells (with rotation) are used.

        Any wells already calibrated are kept, and the fit is redone.
        """
        self._mode_combo.setCurrentText(str(CalibrationMode(mode)))

    def value(self) -> useq.WellPlatePlan | None:
        """Return the plate plan with calibration information."""
        if self._current_plate is None:  # pragma: no cover
            return None
        return useq.WellPlatePlan(
            plate=self._current_plate,
            a1_center_xy=self._a1_center_xy,
            rotation=self._rotation,
        )

    # ---------------------------PRIVATE API----------------------------------

    def _move_to_xy_position(self, pos: useq.Position) -> None:
        """Move the stage to the selected well position."""
        self._mmc.waitForSystem()
        x, y = pos.x, pos.y
        if x is None or y is None:  # pragma: no cover
            return
        self._mmc.setXYPosition(x, y)

    def _move_to_test_position(self) -> None:
        """Move the stage to the edge of the selected well."""
        if well_wdg := self._current_calibration_widget():
            plate = self._current_plate
            if plate is None:  # pragma: no cover
                return
            if well_center := well_wdg.wellCenter():
                rnd_x, rnd_y = self._get_random_edge_point(plate, well_center)
                self._move_to_xy_position(useq.Position(x=rnd_x, y=rnd_y))

    def _get_random_edge_point(
        self, plate: useq.WellPlate, well_center: tuple[float, float]
    ) -> tuple[float, float]:
        """Return a random point along the edge of the well.

        It returns a random point along the circumference of the well if the well is
        circular, otherwise it returns a random point along the edge of the well.
        """
        x, y = well_center
        width = plate.well_size[0] * 1000  # convert to µm
        height = plate.well_size[1] * 1000  # convert to µm

        self._mmc.waitForSystem()
        curr_x, curr_y = self._mmc.getXYPosition()

        while True:
            # if circular, get a random point along the circumference of the well
            if plate.circular_wells:
                angle = np.random.uniform(0, 2 * np.pi)
                rnd_x = x + width / 2 * np.cos(angle)
                rnd_y = y + height / 2 * np.sin(angle)
            # otherwise get the vertices of the squared/rectangular well
            else:
                edges = [
                    (x - width / 2, y - height / 2),  # top left
                    (x + width / 2, y - height / 2),  # top right
                    (x + width / 2, y + height / 2),  # bottom right
                    (x - width / 2, y + height / 2),  # bottom left
                ]
                rnd_x, rnd_y = edges[np.random.randint(0, 4)]
            # make sure the random point is not the current point
            if (round(curr_x), round(curr_y)) != (round(rnd_x), round(rnd_y)):
                return rnd_x, rnd_y

    def _on_mode_changed(self, text: str) -> None:
        """Redo the fit with the newly selected number of calibration wells."""
        self._mode = CalibrationMode(text)
        # keep whatever is already calibrated, just re-solve with the new model
        self._recompute_calibration()

    def _on_tab_changed(self, idx: int) -> None:
        """Hide or show the well calibration widget based on the selected tab."""
        if well_wdg := self._current_calibration_widget():
            well_wdg.setEnabled(idx == 0)  # enable when calibrate tab is selected
        self._test_well_btn.setEnabled(idx == 0)

    def _origin_spacing_rotation(
        self,
    ) -> tuple[tuple[float, float], tuple[float, float], float] | None:
        """Return the origin, scale, and rotation of the plate.

        If the plate is not fully calibrated, returns None.

        The units are a little confusing here, but are chosen to match the units in
        the useq.WellPlatePlan class. The origin is in µm, the well spacing is in mm.

        Returns
        -------
        origin : tuple[float, float]
            The stage coordinates in µm of the center of well A1 (top-left corner).
        well_spacing : tuple[float, float]
            The center-to-center distance in mm (pitch) between wells in the x and y
            directions.
        rotation : float
                a1_center_xy : tuple[float, float]
            The rotation angle in degrees (anti-clockwise) of the plate.
        """
        if len(self._calibrated_wells) < self._mode.wells_required:
            # not enough wells calibrated
            return None

        try:
            if self._mode.solves_rotation:
                params = well_coords_affine(self._calibrated_wells)
            else:
                nominal = None
                if self._current_plate is not None:
                    nominal = (
                        self._current_plate.well_spacing[0] * 1000,
                        self._current_plate.well_spacing[1] * 1000,
                    )
                params = well_coords_affine_no_rotation(self._calibrated_wells, nominal)
        except ValueError:
            # collinear points, or an undetermined spacing
            return None

        a, b, ty, c, d, tx = params
        unit_y = np.hypot(a, c) / 1000  # convert to mm
        unit_x = np.hypot(b, d) / 1000  # convert to mm
        rotation = round(np.rad2deg(np.arctan2(c, a)), 2)

        return (round(tx, 4), round(ty, 4)), (unit_x, unit_y), rotation

    def _get_or_create_well_calibration_widget(
        self, idx: tuple[int, int]
    ) -> WellCalibrationWidget:
        """Create or return the calibration widget for the given well index."""
        if not self._current_plate:  # pragma: no cover
            raise ValueError("No plate set.")
        if idx in self._calibration_widgets:
            return self._calibration_widgets[idx]

        self._calibration_widgets[idx] = wdg = WellCalibrationWidget(self, self._mmc)
        wdg.layout().setContentsMargins(0, 0, 0, 0)

        # set calibration widget well name, and circular well state
        well_name = self._current_plate.all_well_names[idx]
        wdg.well_label.setText(well_name)
        if self._current_plate:
            wdg.setCircularWell(self._current_plate.circular_wells)

        wdg.calibrationChanged.connect(self._on_well_calibration_changed)
        self._calibration_widget_stack.addWidget(wdg)
        return wdg

    def _current_calibration_widget(self) -> WellCalibrationWidget | None:
        return self._calibration_widget_stack.currentWidget()  # type: ignore

    def _on_plate_selection_changed(self) -> None:
        """A well has been selected in the plate view."""
        if not (idx := self._selected_well_index()):
            return

        # create/activate a well calibration widget for the selected well
        with suppress(ValueError):
            well_calib_wdg = self._get_or_create_well_calibration_widget(idx)
            self._calibration_widget_stack.setCurrentWidget(well_calib_wdg)

        # enable/disable test button
        self._test_well_btn.setEnabled(idx in self._calibrated_wells)

    def _on_well_calibration_changed(self, calibrated: bool) -> None:
        """The current well calibration state has been changed."""
        self._test_well_btn.setEnabled(calibrated)
        if idx := self._selected_well_index():
            # update the color of the well in the plate view accordingly
            if calibrated and (well_calib_wdg := self._current_calibration_widget()):
                if center := well_calib_wdg.wellCenter():
                    self._calibrated_wells[idx] = center
                    self._plate_view.setWellColor(*idx, Qt.GlobalColor.green)
            else:
                self._calibrated_wells.pop(idx, None)
                self._plate_view.setWellColor(*idx, None)

        self._recompute_calibration()

    def _recompute_calibration(self) -> None:
        """Re-solve the plate transform from the calibrated wells and report it."""
        osr = self._origin_spacing_rotation()
        if fully_calibrated := (osr is not None):
            self._a1_center_xy, self._well_spacing, self._rotation = osr
            if (plate_plan := self.value()) is not None:
                self._plate_test.drawPlate(plate_plan)
        else:
            self._a1_center_xy = (0.0, 0.0)
            self._rotation = None
            self._well_spacing = None
            self._plate_test.clear()

        self._update_info()
        self._tab_wdg.setTabEnabled(1, fully_calibrated)
        self.calibrationChanged.emit(fully_calibrated)

    def _update_info(self) -> None:
        style = self.style()
        if self._rotation is not None:
            spacing = self._well_spacing or (0, 0)
            txt = "<strong>Plate calibrated.</strong>"
            ico = QIconifyIcon(CALIBRATED_ICON, color=GREEN)
            if self._current_plate is not None:
                spacing_diff = abs(spacing[0] - self._current_plate.well_spacing[0])
                # if spacing is more than 5% different from the plate spacing...
                if spacing_diff > 0.05 * self._current_plate.well_spacing[0]:
                    txt += (
                        "<font color='red'>   Expected well spacing of "
                        f"{self._current_plate.well_spacing[0]:.2f} mm, "
                        f"calibrated at {spacing[0]:.2f}</font>"
                    )
                    ico = style.standardIcon(QStyle.StandardPixmap.SP_MessageBoxWarning)
            txt += "<br>"
            x0, y0 = self._a1_center_xy
            txt += f"\nA1 Center [mm]: ({x0 / 1000:.2f}, {y0 / 1000:.2f}),   "
            txt += f"Well Spacing [mm]: ({spacing[0]:.2f}, {spacing[1]:.2f}),   "
            if self._mode.solves_rotation:
                txt += f"Rotation: {self._rotation}°"
            else:
                txt += "Rotation: assumed 0° (2 well calibration)"
        elif len(self._calibrated_wells) < self._mode.wells_required:
            txt = f"Please calibrate at least {self._mode.wells_required} wells."
            ico = style.standardIcon(QStyle.StandardPixmap.SP_MessageBoxInformation)
        else:
            txt = (
                "Could not calibrate. Ensure points are not collinear "
                "(with 2 wells, that they lie in different rows and columns)."
            )
            ico = style.standardIcon(QStyle.StandardPixmap.SP_MessageBoxWarning)
        self._info_icon.setPixmap(ico.pixmap(42))
        self._info.setText(txt)

    def _selected_well_index(self) -> tuple[int, int] | None:
        if selected := self._plate_view.selectedIndices():
            return selected[0]
        return None


def well_coords_affine_no_rotation(
    index_coordinates: Mapping[tuple[int, int], tuple[float, float]],
    nominal_spacing: tuple[float, float] | None = None,
) -> tuple[float, float, float, float, float, float]:
    """Return the best-fit *axis aligned* transform mapping indices to coordinates.

    Like `well_coords_affine`, but with the rotation constrained to zero, so that
    two calibrated wells are enough (a rotation needs three).  The x and y
    equations then decouple: x depends only on the column, y only on the row.

    Parameters
    ----------
    index_coordinates : Mapping[tuple[int, int], tuple[float, float]]
        A mapping of grid indices to world coordinates.
    nominal_spacing : tuple[float, float] | None
        The (x, y) well spacing in µm to fall back on for an axis along which
        the calibrated wells do not differ (e.g. two wells in the same row say
        nothing about the row spacing).  If None, such a case is an error.

    Returns
    -------
    tuple: the same six parameters returned by `well_coords_affine`, with the
    off-diagonal (rotation/shear) terms set to zero.
    """
    rows, cols, xs, ys = [], [], [], []
    for (row, col), (x, y) in index_coordinates.items():
        rows.append(row)
        cols.append(col)
        xs.append(x)
        ys.append(y)

    def _fit(indices: list[int], values: list[float], fallback: float | None) -> tuple:
        """Fit value = scale * index + offset, over the given points."""
        idx = np.asarray(indices, dtype=float)
        val = np.asarray(values, dtype=float)
        if np.ptp(idx) > 0:
            scale, offset = np.linalg.lstsq(
                np.column_stack([idx, np.ones_like(idx)]), val, rcond=None
            )[0]
            return float(scale), float(offset)
        # all points share the same index: the spacing cannot be determined
        if fallback is None:
            raise ValueError(
                "Cannot determine the well spacing. Calibrate wells in "
                "different rows and columns."
            )
        return fallback, float(np.mean(val))

    nom_x, nom_y = nominal_spacing if nominal_spacing is not None else (None, None)
    # well plate indices go up in row as we go down in y, hence the negated row
    d, tx = _fit(cols, xs, nom_x)
    a, ty = _fit([-r for r in rows], ys, nom_y)
    return (a, 0.0, ty, 0.0, d, tx)


def well_coords_affine(
    index_coordinates: Mapping[tuple[int, int], tuple[float, float]],
) -> tuple[float, float, float, float, float, float]:
    """Return best-fit transformation that maps well plate indices to world coordinates.

    Parameters
    ----------
    index_coordinates : Mapping[tuple[int, int], tuple[float, float]]
        A mapping of grid indices to world coordinates.

    Returns
    -------
    np.ndarray: The affine transformation matrix of shape (3, 3).
    """
    A: list[list[int]] = []
    B: list[float] = []
    for (row, col), (x, y) in index_coordinates.items():
        # well plate indices go up in row as we go down in y
        # so we have to negate the row to get the correct transformation
        A.append([-row, col, 1, 0, 0, 0])
        A.append([0, 0, 0, -row, col, 1])
        # row corresponds to y, col corresponds to x
        B.extend((y, x))

    # Solve the least squares problem to find the affine transformation parameters
    params, _, rank, _ = np.linalg.lstsq(np.array(A), np.array(B), rcond=None)

    if rank != 6:
        raise ValueError("Underdetermined system of equations. Are points collinear?")

    return tuple(params)
