from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from pymmcore_widgets import DefaultCameraExposureWidget

if TYPE_CHECKING:
    from pymmcore_plus import CMMCorePlus
    from pytestqt.qtbot import QtBot


def test_exposure_widget(qtbot: QtBot, global_mmcore: CMMCorePlus):
    global_mmcore.setExposure(15)
    wdg = DefaultCameraExposureWidget(mmcore=global_mmcore)
    qtbot.addWidget(wdg)

    # check that it get's whatever core is set to.
    assert wdg.spinBox.value() == 15
    global_mmcore.setExposure(30)
    qtbot.waitUntil(lambda: wdg.spinBox.value() == 30)

    wdg.spinBox.setValue(45)
    qtbot.waitUntil(lambda: global_mmcore.getExposure() == 45)

    # test updating cameraDevice
    global_mmcore.setProperty("Core", "Camera", "")
    assert not wdg.isEnabled()

    with pytest.raises(RuntimeError):
        wdg.setCamera("blarg")

    # set to an invalid camera name
    # should now be disabled.
    wdg.setCamera("blarg", force=True)
    assert not wdg.isEnabled()

    # reset the camera to a working one
    global_mmcore.setProperty("Core", "Camera", "Camera")
    wdg.spinBox.setValue(0.1)
    qtbot.waitUntil(lambda: global_mmcore.getExposure() == 0.1)


def test_exposure_widget_cross_thread_update(qtbot: QtBot, global_mmcore: CMMCorePlus):
    """Core events delivered on a worker thread must not echo setExposure.

    With the psygnal backend, core events run handlers on the emitting (e.g.
    acquisition) thread. The spinbox update must be marshalled to the main
    thread inside the signals_blocked guard; otherwise the spinbox can emit
    valueChanged (wired to mmc.setExposure) with a stale value and override
    the exposure of a frame in a running acquisition.
    """
    import threading

    global_mmcore.setExposure(20)
    wdg = DefaultCameraExposureWidget(mmcore=global_mmcore)
    qtbot.addWidget(wdg)
    assert wdg.spinBox.value() == 20

    echoes: list[float] = []
    wdg.spinBox.valueChanged.connect(echoes.append)

    def emit_from_thread() -> None:
        wdg._on_exp_changed("Camera", 33.0)

    t = threading.Thread(target=emit_from_thread)
    t.start()
    t.join()

    # the update must be QUEUED to the main thread, not applied on the
    # worker: a synchronous cross-thread setValue is undefined behavior and
    # escapes the signals_blocked guard
    assert wdg.spinBox.value() == 20
    qtbot.waitUntil(lambda: wdg.spinBox.value() == 33.0)
    assert echoes == []
