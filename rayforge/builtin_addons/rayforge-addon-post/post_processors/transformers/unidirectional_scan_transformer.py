from __future__ import annotations

import math
from gettext import gettext as _
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from raygeo.ops import Ops
from raygeo.ops.types import CommandCategory, CommandType

from rayforge.pipeline.transformer.base import ExecutionPhase, OpsTransformer
from rayforge.shared.tasker.progress import ProgressContext

if TYPE_CHECKING:
    from raygeo.geo import Geometry

    from rayforge.core.workpiece import WorkPiece


class UnidirectionalScanTransformer(OpsTransformer):
    """
    Forces every raster scan pass to travel in the same geometric
    direction, instead of the default alternating (serpentine/zigzag)
    pattern.

    Standard raster engraving alternates direction every line to
    minimize travel time. On machines where the scan axis has
    mechanical backlash (most commonly a rotary attachment driving one
    of the axes), that alternation makes every other line land at a
    slightly different position, producing visible misalignment.

    This transformer walks the generated raster passes and, for every
    pass that runs opposite to the direction of the first pass, rewrites
    it: it repositions (via a non-cutting rapid move) to the far end of
    the line and re-emits the scan in the forward direction, with the
    power samples reversed so the same pixels are still burned at the
    same physical location. This costs one extra rapid move per
    reversed line, in exchange for the scan axis never changing
    direction.
    """

    def __init__(self, enabled: bool = False):
        super().__init__(enabled=enabled)

    @property
    def execution_phase(self) -> ExecutionPhase:
        return ExecutionPhase.POST_PROCESSING

    @property
    def label(self) -> str:
        return _("Unidirectional Scan")

    @property
    def description(self) -> str:
        return _(
            "Forces all raster passes to scan in the same direction, "
            "avoiding backlash artifacts from direction reversal "
            "(e.g. on a rotary axis)."
        )

    def run(
        self,
        ops: Ops,
        workpiece: Optional["WorkPiece"] = None,
        context: Optional[ProgressContext] = None,
        stock_geometries: Optional[List["Geometry"]] = None,
        settings: Optional[Dict[str, Any]] = None,
    ) -> None:
        unidirectional = (
            settings.get("unidirectional_scan", False) if settings else False
        )
        if not self.enabled or not unidirectional:
            return

        source = ops.copy()
        ops.clear()
        n = source.len()
        idx = 0
        forward_dir: Optional[tuple] = None
        while idx < n:
            if source.command_type(idx) == CommandType.MOVE_TO:
                move_end = source.endpoint(idx)
                j = idx + 1
                while j < n and source.category(j) == CommandCategory.STATE:
                    j += 1
                if j < n and source.is_scanline(j):
                    scan_end = source.endpoint(j)
                    dx = scan_end[0] - move_end[0]
                    dy = scan_end[1] - move_end[1]

                    if forward_dir is None and (
                        not math.isclose(dx, 0.0, abs_tol=1e-9)
                        or not math.isclose(dy, 0.0, abs_tol=1e-9)
                    ):
                        forward_dir = (dx, dy)

                    dot = (
                        dx * forward_dir[0] + dy * forward_dir[1]
                        if forward_dir is not None
                        else 1.0
                    )
                    if forward_dir is not None and dot < 0:
                        power = list(source.scanline_data(j))[::-1]
                        ops.move_to(
                            scan_end[0],
                            scan_end[1],
                            scan_end[2],
                            extra=source.extra_axes(j),
                        )
                        for k in range(idx + 1, j):
                            ops.transfer_command_from(source, k)
                        ops.scan_to(
                            move_end[0],
                            move_end[1],
                            move_end[2],
                            power_values=power,
                            extra=source.extra_axes(idx),
                        )
                    else:
                        for k in range(idx, j + 1):
                            ops.transfer_command_from(source, k)
                    idx = j + 1
                    continue
            ops.transfer_command_from(source, idx)
            idx += 1

    def to_dict(self) -> Dict[str, Any]:
        return {**super().to_dict()}

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any]
    ) -> "UnidirectionalScanTransformer":
        return cls(enabled=data.get("enabled", False))
