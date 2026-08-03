"""Example usage of the StageMapWidget class.

A live, zoomable map of the XY stage: select a well plate format from the
toolbar dropdown to overlay its outline, calibrate it once (the calibration is
stored on disk and reloaded next session), and double-click anywhere on the map
to move the stage there.

The map has no position list of its own: it binds to one that already exists
in the application, here the positions tab of an `MDAWidget`.  Positions added
there show up on the map right away, joined by arrows in acquisition order, and
"Assign Positions to Nearest Well" fills in the "Well" column of the table,
which is stored as plate_row/plate_col and becomes the plate/well hierarchy of
an OME-Zarr dataset.  Names stay free for your own labels.
"""

from pymmcore_plus import CMMCorePlus
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication, QSplitter

from pymmcore_widgets import MDAWidget, StageMapWidget

app = QApplication([])

mmc = CMMCorePlus().instance()
mmc.loadSystemConfiguration()

mda = MDAWidget()
mda.setValue(mda.value().replace(stage_positions=[{"x": 0, "y": 0, "name": "start"}]))

stage_map = StageMapWidget()
stage_map.setPositionTable(mda.stage_positions)

splitter = QSplitter(Qt.Orientation.Horizontal)
splitter.addWidget(mda)
splitter.addWidget(stage_map)
splitter.setStretchFactor(1, 2)
splitter.resize(1300, 650)
splitter.show()

app.exec_()
