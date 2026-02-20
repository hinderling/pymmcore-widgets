from __future__ import annotations

from pymmcore_plus import CMMCorePlus
from qtpy.QtCore import Qt, QThread, Signal
from qtpy.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QWidget,
)

from pymmcore_widgets._util import load_system_config


class _CfgLoadThread(QThread):
    """Worker: run load_system_config off the main thread."""

    failed = Signal(str)

    def __init__(self, path: str, mmc: CMMCorePlus, parent: QWidget | None = None):
        super().__init__(parent)
        self._path = path
        self._mmc = mmc

    def run(self) -> None:
        try:
            load_system_config(self._path, self._mmc)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


class ConfigurationWidget(QWidget):
    """A Widget to select and load a micromanager system configuration.

    Parameters
    ----------
    parent : QWidget | None
        Optional parent widget. By default, None.
    mmcore : CMMCorePlus | None
        Optional [`pymmcore_plus.CMMCorePlus`][] micromanager core.
        By default, None. If not specified, the widget will use the active
        (or create a new)
        [`CMMCorePlus.instance`][pymmcore_plus.core._mmcore_plus.CMMCorePlus.instance].
    """

    def __init__(
        self,
        *,
        parent: QWidget | None = None,
        mmcore: CMMCorePlus | None = None,
    ) -> None:
        super().__init__(parent=parent)

        self._mmc = mmcore or CMMCorePlus.instance()

        self.cfg_LineEdit = QLineEdit()
        self.cfg_LineEdit.setPlaceholderText("MMConfig_demo.cfg")

        self.browse_cfg_Button = QPushButton("...")
        self.browse_cfg_Button.clicked.connect(self._browse_cfg)

        self.load_cfg_Button = QPushButton("Load")
        self.load_cfg_Button.clicked.connect(self._load_cfg)

        self.setLayout(QHBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().addWidget(self.cfg_LineEdit)
        self.layout().addWidget(self.browse_cfg_Button)
        self.layout().addWidget(self.load_cfg_Button)

    def _browse_cfg(self) -> None:
        """Open file dialog to select a config file."""
        (filename, _) = QFileDialog.getOpenFileName(
            self, "Select a Micro-Manager configuration file", "", "cfg(*.cfg)"
        )
        if filename:
            self.cfg_LineEdit.setText(filename)

    def _load_cfg(self) -> None:
        """Load the config path currently in the line_edit, off the main thread.

        A modal progress dialog is shown while loading to keep the GUI responsive.
        """
        path = self.cfg_LineEdit.text().strip()

        dlg = QProgressDialog("Loading configuration…", None, 0, 0, self)
        dlg.setWindowTitle("Loading")
        dlg.setWindowModality(Qt.WindowModality.ApplicationModal)
        dlg.setMinimumDuration(0)
        dlg.setValue(0)
        dlg.show()
        self.load_cfg_Button.setEnabled(False)

        def _on_done() -> None:
            dlg.close()
            self.load_cfg_Button.setEnabled(True)

        def _on_failed(msg: str) -> None:
            dlg.close()
            self.load_cfg_Button.setEnabled(True)
            QMessageBox.critical(self, "Configuration Load Error", msg)

        self._load_thread = _CfgLoadThread(path, self._mmc, parent=self)
        self._load_thread.finished.connect(_on_done)
        self._load_thread.failed.connect(_on_failed)
        self._load_thread.start()

    def setTitle(self, title: str) -> None:
        _show_deprecation("setTitle")

    def title(self) -> str:
        _show_deprecation("title")
        return ""


def _show_deprecation(name: str) -> None:
    import warnings

    warnings.warn(
        "ConfigurationWidget is no longer a QGroupBox. "
        f"Please place it in a groupbox if you need {name}",
        DeprecationWarning,
        stacklevel=3,
    )
