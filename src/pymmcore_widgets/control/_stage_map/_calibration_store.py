"""Persistence of well-plate calibrations (plate <-> stage alignment).

A plate calibration (the stage coordinates of well A1 and the rotation of the
plate relative to the stage axes) typically does not change between
experiments, so it is worth persisting it to disk and reloading it in the next
session.  This module stores one JSON file per plate name (a serialized
[useq.WellPlatePlan][]) in a user-data directory.
"""

from __future__ import annotations

import re
from pathlib import Path

import useq
from qtpy.QtCore import QStandardPaths

__all__ = ["PlateCalibrationStore"]


def _default_calibration_dir() -> Path:
    """Return the default directory used to store plate calibrations."""
    # GenericDataLocation is independent of the QApplication name, so the same
    # calibrations are found regardless of which application hosts the widget.
    base = QStandardPaths.writableLocation(
        QStandardPaths.StandardLocation.GenericDataLocation
    )
    root = Path(base) if base else Path.home() / ".local" / "share"
    return root / "pymmcore-widgets" / "plate_calibrations"


def _slugify(name: str) -> str:
    """Return a filesystem-safe version of a plate name."""
    return re.sub(r"[^\w\-.]", "_", name.strip()) or "unnamed"


class PlateCalibrationStore:
    """Store/retrieve plate calibrations as JSON files, one per plate name.

    Parameters
    ----------
    directory : Path | str | None
        Directory in which to store the calibration files.  By default, a
        "pymmcore-widgets/plate_calibrations" folder inside the user's
        standard data location (e.g. ``~/Library/Application Support`` on
        macOS, ``~/.local/share`` on Linux).  The directory is only created
        when a calibration is first saved.
    """

    def __init__(self, directory: Path | str | None = None) -> None:
        self._dir = Path(directory) if directory is not None else None

    @property
    def directory(self) -> Path:
        """The directory in which calibration files are stored."""
        return self._dir if self._dir is not None else _default_calibration_dir()

    def path_for(self, plate_name: str) -> Path:
        """Return the path of the calibration file for `plate_name`."""
        return self.directory / f"{_slugify(plate_name)}.json"

    def has_calibration(self, plate_name: str) -> bool:
        """Return True if a calibration is stored for `plate_name`."""
        return self.path_for(plate_name).is_file()

    def save(self, plan: useq.WellPlatePlan) -> Path:
        """Persist the calibration of `plan`, returning the file written.

        Only the plate definition, the A1 center and the rotation are stored
        (any well selection or points plan on `plan` is intentionally
        discarded, since those are experiment-specific).
        """
        plate = plan.plate
        minimal = useq.WellPlatePlan(
            plate=plate, a1_center_xy=plan.a1_center_xy, rotation=plan.rotation
        )
        dest = self.path_for(plate.name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(minimal.model_dump_json(exclude_unset=True, indent=2))
        return dest

    def load(self, plate_name: str) -> useq.WellPlatePlan | None:
        """Return the stored calibration for `plate_name`, or None."""
        file = self.path_for(plate_name)
        return self._read(file) if file.is_file() else None

    def delete(self, plate_name: str) -> bool:
        """Delete the stored calibration for `plate_name`.

        Returns True if a file was deleted, False if none existed.
        """
        file = self.path_for(plate_name)
        if file.is_file():
            file.unlink()
            return True
        return False

    @staticmethod
    def _read(file: Path) -> useq.WellPlatePlan | None:
        try:
            return useq.WellPlatePlan.model_validate_json(file.read_text())
        except (ValueError, OSError):  # pragma: no cover
            return None
