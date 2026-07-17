import logging
import math
from gettext import gettext as _
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from raygeo.ops import Ops
from raygeo.ops.types import CommandCategory, CommandType

from rayforge.core.workpiece import WorkPiece
from rayforge.pipeline.transformer.base import ExecutionPhase, OpsTransformer
from rayforge.shared.tasker.progress import ProgressContext

if TYPE_CHECKING:
    from raygeo.geo import Geometry

logger = logging.getLogger(__name__)


class UnidirectionalScanTransformer(OpsTransformer):
    """
    Forces every raster scan pass in the fully-assembled ops to travel
    in the same geometric direction, and reorders passes so the axis
    perpendicular to the scan direction only ever moves one way.

    Standard raster engraving alternates direction every line, and
    large images are generated in chunks - each wrapped in its own
    ops-section markers - whose passes can end up out of order
    relative to each other. Both make a backlash-sensitive axis (most
    commonly a rotary attachment) reverse direction: reversing a
    single line's direction is not enough on its own if whole chunks
    are still visited out of sequence.

    This collects every raster pass belonging to the same workpiece
    (across all of its chunk sections), aligns each to a shared
    forward direction (reversing the power samples of any pass that
    runs the other way, so the same physical points still get the
    same power), and sorts them all by position along the
    perpendicular axis - so the constrained axis only ever advances
    for the whole workpiece, not just within one chunk. The merged
    passes are re-wrapped in a single section marker equivalent to the
    ones that were merged.

    Non-pass, non-section commands (e.g. workpiece boundaries) are
    left in place and still separate distinct reorder groups - this
    does not reorder across different workpieces.
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
            "Forces all raster passes - and the order they run in - to "
            "only ever advance along one direction, avoiding backlash "
            "from axis reversal (e.g. on a rotary axis)."
        )

    def run(
        self,
        ops: Ops,
        workpiece: Optional[WorkPiece] = None,
        context: Optional[ProgressContext] = None,
        stock_geometries: Optional[List["Geometry"]] = None,
        settings: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.enabled:
            return
        _force_unidirectional_scan_and_sort(ops)

    def to_dict(self) -> Dict[str, Any]:
        return {**super().to_dict()}

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any]
    ) -> "UnidirectionalScanTransformer":
        return cls(enabled=data.get("enabled", False))


def _force_unidirectional_scan_and_sort(ops: Ops) -> None:
    source = ops.copy()
    n = source.len()

    # result_seq holds, in order, either ('fixed', idx) for a command
    # that isn't part of a pass or a chunk-section marker, or
    # ('run', run_dict) for a mergeable group of raster passes.
    # run_dict = {'passes': [...], 'section': params or None,
    #             'head': head_uid or None}
    result_seq: List[Tuple[str, Any]] = []
    current_run: Optional[Dict[str, Any]] = None
    pending_section = None
    pending_head = None
    forward_dir: Optional[Tuple[float, float]] = None

    idx = 0
    while idx < n:
        ct = source.command_type(idx)

        if ct == CommandType.OPS_SECTION_START:
            if pending_section is None:
                pending_section = source.section_params(idx)
            idx += 1
            continue
        if ct == CommandType.OPS_SECTION_END:
            idx += 1
            continue
        if ct == CommandType.SET_HEAD:
            if pending_head is None:
                pending_head = source.head_uid(idx)
            idx += 1
            continue

        if ct == CommandType.MOVE_TO:
            move_end = source.endpoint(idx)
            j = idx + 1
            state_idxs = []
            while j < n and source.category(j) == CommandCategory.STATE:
                state_idxs.append(j)
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

                if current_run is None:
                    current_run = {
                        "passes": [],
                        "section": pending_section,
                        "head": pending_head,
                    }
                    pending_section = None
                    pending_head = None
                    result_seq.append(("run", current_run))

                current_run["passes"].append(
                    {
                        "start": move_end,
                        "end": scan_end,
                        "power": list(source.scanline_data(j)),
                        "state_idxs": state_idxs,
                        "move_extra": source.extra_axes(idx),
                        "scan_extra": source.extra_axes(j),
                    }
                )
                idx = j + 1
                continue

        current_run = None
        pending_section = None
        pending_head = None
        result_seq.append(("fixed", idx))
        idx += 1

    if forward_dir is None:
        return

    perp = (-forward_dir[1], forward_dir[0])

    def pass_key(p: Dict[str, Any]) -> float:
        return p["start"][0] * perp[0] + p["start"][1] * perp[1]

    def align(p: Dict[str, Any]) -> None:
        dx = p["end"][0] - p["start"][0]
        dy = p["end"][1] - p["start"][1]
        dot = dx * forward_dir[0] + dy * forward_dir[1]
        if dot < 0:
            p["start"], p["end"] = p["end"], p["start"]
            p["move_extra"], p["scan_extra"] = (
                p["scan_extra"],
                p["move_extra"],
            )
            p["power"] = p["power"][::-1]

    for kind, val in result_seq:
        if kind == "run":
            for p in val["passes"]:
                align(p)
            val["passes"].sort(key=pass_key)

    ops.clear()
    for kind, val in result_seq:
        if kind == "fixed":
            ops.transfer_command_from(source, val)
            continue

        run = val
        if run["head"]:
            ops.set_head(run["head"])
        if run["section"]:
            section_type, wp_uid, raster_mode = run["section"]
            ops.ops_section_start(
                section_type, wp_uid, raster_mode=raster_mode
            )
        for p in run["passes"]:
            ops.move_to(
                p["start"][0],
                p["start"][1],
                p["start"][2],
                extra=p["move_extra"],
            )
            for sidx in p["state_idxs"]:
                ops.transfer_command_from(source, sidx)
            ops.scan_to(
                p["end"][0],
                p["end"][1],
                p["end"][2],
                power_values=p["power"],
                extra=p["scan_extra"],
            )
        if run["section"]:
            section_type, _wp_uid, raster_mode = run["section"]
            ops.ops_section_end(section_type, raster_mode=raster_mode)
