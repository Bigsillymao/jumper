"""jumper.dance_feiyi -- 《螃蟹学非遗》: the Huagudeng slide-step (闪身步) and
the Guzi-yangge tremble (狗熊哆嗦毛), authored for the 2026 crab-robot challenge.

Both moves are national intangible heritage dances that went viral on Douyin in
October 2026. The piece is authored by `tools/author_feiyi_dance.py` (submission
repository): four 闪身步 cycles (whip a claw out, twist, slide, freeze), then
the 哆嗦毛 section — a 7 Hz whole-body tremble riding a "隆咚哐" drum pattern
with the weight sinking on the heavy beat — then the bow.

The environment is `tasks/jumper/common/dance/env.py`; this directory adds the
clip and its numbers, following `jumper.dance_dazhaxie`'s conventions.
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.dance_feiyi",
    assets=JUMPER_ASSETS,
    description="jumper hexapod dancing the authored folk-dance piece "
                "(闪身步 + 狗熊哆嗦毛); the clip is committed in "
                "tasks/jumper/dance_feiyi/media/",
    tags=("imitation", "jumper", "dance", "flat", "authored", "folk"),
)
