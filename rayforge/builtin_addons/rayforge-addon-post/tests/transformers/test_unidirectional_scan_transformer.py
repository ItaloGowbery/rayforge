import pytest
from post_processors.transformers import UnidirectionalScanTransformer
from raygeo.ops import Ops
from raygeo.ops.types import CommandType

from rayforge.pipeline.transformer.base import ExecutionPhase


@pytest.fixture
def transformer() -> UnidirectionalScanTransformer:
    """Provides a default, enabled UnidirectionalScanTransformer instance."""
    return UnidirectionalScanTransformer(enabled=True)


def _build_zigzag_x() -> Ops:
    """Three-row raster scanning along X: row0 LTR, row1 RTL, row2 LTR."""
    ops = Ops()
    ops.move_to(0.0, 0.6, 0.0)
    ops.scan_to(2.0, 0.6, 0.0, power_values=[200, 200, 200, 200])
    ops.move_to(2.0, 0.5, 0.0)
    ops.scan_to(0.0, 0.5, 0.0, power_values=[210, 220, 230, 240])
    ops.move_to(0.0, 0.4, 0.0)
    ops.scan_to(2.0, 0.4, 0.0, power_values=[220, 220, 220, 220])
    return ops


def _build_zigzag_y() -> Ops:
    """Three-column raster scanning along Y (e.g. a rotary axis):
    col0 goes 0->4, col1 goes 4->0, col2 goes 0->4."""
    ops = Ops()
    ops.move_to(0.0, 0.0, 0.0)
    ops.scan_to(0.0, 4.0, 0.0, power_values=[10, 20, 30, 40])
    ops.move_to(0.5, 4.0, 0.0)
    ops.scan_to(0.5, 0.0, 0.0, power_values=[50, 60, 70, 80])
    ops.move_to(1.0, 0.0, 0.0)
    ops.scan_to(1.0, 4.0, 0.0, power_values=[90, 100, 110, 120])
    return ops


def test_execution_phase_is_correct(transformer: UnidirectionalScanTransformer):
    assert transformer.execution_phase == ExecutionPhase.POST_PROCESSING


def test_serialization_and_deserialization():
    original = UnidirectionalScanTransformer(enabled=False)
    data = original.to_dict()
    recreated = UnidirectionalScanTransformer.from_dict(data)
    assert data["name"] == "UnidirectionalScanTransformer"
    assert data["enabled"] is False
    assert isinstance(recreated, UnidirectionalScanTransformer)
    assert recreated.enabled is False


def test_no_op_when_disabled():
    ops = _build_zigzag_x()
    original = [ops.endpoint(i) for i in range(ops.len())]

    transformer = UnidirectionalScanTransformer(enabled=False)
    transformer.run(ops, settings={"unidirectional_scan": True})

    assert [ops.endpoint(i) for i in range(ops.len())] == original


def test_no_op_when_flag_off(transformer: UnidirectionalScanTransformer):
    ops = _build_zigzag_x()
    original = [ops.endpoint(i) for i in range(ops.len())]

    transformer.run(ops, settings={"unidirectional_scan": False})

    assert [ops.endpoint(i) for i in range(ops.len())] == original


def test_no_op_without_settings(transformer: UnidirectionalScanTransformer):
    ops = _build_zigzag_x()
    original = [ops.endpoint(i) for i in range(ops.len())]

    transformer.run(ops, settings=None)

    assert [ops.endpoint(i) for i in range(ops.len())] == original


def test_forces_same_direction_on_x(
    transformer: UnidirectionalScanTransformer,
):
    ops = _build_zigzag_x()

    transformer.run(ops, settings={"unidirectional_scan": True})

    assert ops.len() == 6
    # Row 0 (already LTR): untouched.
    assert ops.endpoint(0) == pytest.approx((0.0, 0.6, 0.0))
    assert ops.endpoint(1) == pytest.approx((2.0, 0.6, 0.0))
    assert list(ops.scanline_data(1)) == [200, 200, 200, 200]
    # Row 1 (was RTL): reversed to LTR, power samples reversed too.
    assert ops.command_type(2) == CommandType.MOVE_TO
    assert ops.endpoint(2) == pytest.approx((0.0, 0.5, 0.0))
    assert ops.command_type(3) == CommandType.SCAN_LINE
    assert ops.endpoint(3) == pytest.approx((2.0, 0.5, 0.0))
    assert list(ops.scanline_data(3)) == [240, 230, 220, 210]
    # Row 2 (already LTR): untouched.
    assert ops.endpoint(4) == pytest.approx((0.0, 0.4, 0.0))
    assert ops.endpoint(5) == pytest.approx((2.0, 0.4, 0.0))
    assert list(ops.scanline_data(5)) == [220, 220, 220, 220]


def test_forces_same_direction_on_y_axis(
    transformer: UnidirectionalScanTransformer,
):
    """The scan axis reversing doesn't have to be X - this is the case
    that matters for a rotary attachment driving the Y axis."""
    ops = _build_zigzag_y()

    transformer.run(ops, settings={"unidirectional_scan": True})

    for i in (0, 2, 4):
        assert ops.command_type(i) == CommandType.MOVE_TO
    for i in (1, 3, 5):
        assert ops.command_type(i) == CommandType.SCAN_LINE

    # Every pass now goes from Y=0.0 to Y=4.0, never the reverse.
    assert ops.endpoint(0)[1] == pytest.approx(0.0)
    assert ops.endpoint(1)[1] == pytest.approx(4.0)
    assert ops.endpoint(2)[1] == pytest.approx(0.0)
    assert ops.endpoint(3)[1] == pytest.approx(4.0)
    assert ops.endpoint(4)[1] == pytest.approx(0.0)
    assert ops.endpoint(5)[1] == pytest.approx(4.0)

    assert list(ops.scanline_data(1)) == [10, 20, 30, 40]
    # Column 1 was originally 4->0 with [50, 60, 70, 80]; reversed to
    # 0->4 the samples must reverse too, so the same physical points
    # keep the same power.
    assert list(ops.scanline_data(3)) == [80, 70, 60, 50]
    assert list(ops.scanline_data(5)) == [90, 100, 110, 120]


def test_preserves_intermediate_state_commands(
    transformer: UnidirectionalScanTransformer,
):
    """A SetPower between the entry MoveTo and the ScanLine must survive,
    for both a forward and a reversed pass."""
    ops = Ops()
    ops.move_to(0.0, 0.0, 0.0)
    ops.set_power(0.8)
    ops.scan_to(2.0, 0.0, 0.0, power_values=[1, 2, 3, 4])
    ops.move_to(2.0, 1.0, 0.0)
    ops.set_power(0.5)
    ops.scan_to(0.0, 1.0, 0.0, power_values=[5, 6, 7, 8])

    transformer.run(ops, settings={"unidirectional_scan": True})

    assert ops.len() == 6
    assert ops.command_type(1) == CommandType.SET_POWER
    assert ops.power(1) == pytest.approx(0.8)
    assert ops.command_type(4) == CommandType.SET_POWER
    assert ops.power(4) == pytest.approx(0.5)
    # The reversed pass still ends up going 0.0 -> 2.0 in X.
    assert ops.endpoint(3) == pytest.approx((0.0, 1.0, 0.0))
    assert ops.endpoint(5) == pytest.approx((2.0, 1.0, 0.0))
    assert list(ops.scanline_data(5)) == [8, 7, 6, 5]
