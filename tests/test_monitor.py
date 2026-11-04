from cabinsense.monitor import DriverMonitor, FrameSignals, MonitorConfig

FPS = 30


def run(monitor, seconds, start=0.0, **sig):
    events = []
    for i in range(int(seconds * FPS)):
        events += monitor.update(FrameSignals(t=start + i / FPS, **sig))
    return events


def kinds(events):
    return [e.kind for e in events]


def test_alert_free_when_attentive():
    m = DriverMonitor()
    assert run(m, 60, face=True, ear=0.3, mar=0.1) == []


def test_microsleep_after_long_eye_closure():
    m = DriverMonitor()
    run(m, 5, face=True, ear=0.3)
    ev = run(m, 2, start=5, face=True, ear=0.1)
    assert "microsleep" in kinds(ev)


def test_short_blink_is_not_microsleep():
    m = DriverMonitor()
    ev = run(m, 0.3, face=True, ear=0.1) + run(m, 5, start=0.3, face=True, ear=0.3)
    assert "microsleep" not in kinds(ev)


def test_drowsiness_from_perclos():
    m = DriverMonitor(MonitorConfig(perclos_window_s=10, eyes_closed_alert_s=99))
    ev = []
    for k in range(20):  # 0.4 s closed every 1 s => PERCLOS ~40%
        ev += run(m, 0.6, start=k, face=True, ear=0.3) + run(m, 0.4, start=k + 0.6, face=True, ear=0.1)
    assert "drowsiness" in kinds(ev)


def test_yawn_detection():
    m = DriverMonitor()
    assert "yawn" in kinds(run(m, 1.5, face=True, ear=0.3, mar=0.8))


def test_distraction_needs_sustained_head_turn():
    m = DriverMonitor()
    run(m, 4, face=True, ear=0.3, yaw=5, pitch=20)  # calibration: this driver's neutral pose
    assert m.neutral == (5, 20)
    assert "distraction" not in kinds(run(m, 1.0, start=4, face=True, ear=0.3, yaw=50, pitch=20))
    assert "distraction" in kinds(run(m, 2.5, start=5, face=True, ear=0.3, yaw=50, pitch=20))


def test_narrow_eyes_are_not_closed_after_calibration():
    m = DriverMonitor()
    # A driver whose open-eye EAR is naturally 0.19 (below the generic 0.21 cut-off).
    ev = run(m, 20, face=True, ear=0.19)
    assert abs(m.ear_threshold - 0.133) < 1e-3
    assert "drowsiness" not in kinds(ev) and "microsleep" not in kinds(ev)


def test_neutral_offset_is_not_distraction():
    m = DriverMonitor()
    # Camera mounted high: the driver always reads pitch +22, which must not raise alerts.
    assert "distraction" not in kinds(run(m, 20, face=True, ear=0.3, pitch=22))


def test_phone_use_and_cooldown():
    m = DriverMonitor(MonitorConfig(cooldown_s=5))
    ev = run(m, 4, face=True, ear=0.3, phone=True)
    assert kinds(ev).count("phone_use") == 1  # cooldown suppresses repeats


def test_no_face_alert():
    m = DriverMonitor()
    assert "no_face" in kinds(run(m, 3, face=False))
