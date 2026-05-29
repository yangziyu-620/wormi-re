"""VirtualHome constants ported from TMoW (ICLR 2026).

Source: github.com/doldam0/tmow `tmow/utils/virtualhome/const.py`.
The 78-task list, the seen-task stratified index, and the canonical 4-room
layout are taken verbatim. House ID lists are kept for reference but the
WorMI reconstruction maps them to local init-graph cache keys at build time
(see tools/build_virtualhome_dataset_tmow.py).
"""

from __future__ import annotations

# fmt: off
TASKS_SET: dict[str, list[tuple[str, str, str]]] = {
    "Turn on tv":                    [("tv", "is", "on")],
    "Turn on radio":                 [("radio", "is", "on")],
    "Turn on microwave":             [("microwave", "is", "on")],
    "Turn on stove":                 [("stove", "is", "on")],
    "Turn on computer":              [("computer", "is", "on")],
    "Turn on coffeemaker":           [("coffeemaker", "is", "on")],
    "Turn on dishwasher":            [("dishwasher", "is", "on")],
    "Turn on faucet":                [("faucet", "is", "on")],
    "Turn on toaster":               [("toaster", "is", "on")],

    "Open cabinet":                  [("cabinet", "is", "open")],
    "Open dishwasher":               [("dishwasher", "is", "open")],
    "Open microwave":                [("microwave", "is", "open")],
    "Open stove":                    [("stove", "is", "open")],
    "Open coffeepot":                [("coffeepot", "is", "open")],
    "Open toilet":                   [("toilet", "is", "open")],
    "Open fridge":                   [("fridge", "is", "open")],

    "Put apple on desk":             [("apple", "on", "desk")],
    "Put remotecontrol on desk":     [("remotecontrol", "on", "desk")],
    "Put notes on desk":             [("notes", "on", "desk")],
    "Put book on desk":              [("book", "on", "desk")],
    "Put mug on desk":               [("mug", "on", "desk")],
    "Put book on sofa":              [("book", "on", "sofa")],
    "Put chips on sofa":             [("chips", "on", "sofa")],
    "Put clock on sofa":             [("clock", "on", "sofa")],
    "Put folder on sofa":            [("folder", "on", "sofa")],
    "Put milk on sofa":              [("milk", "on", "sofa")],
    "Put cereal on coffeetable":     [("cereal", "on", "coffeetable")],
    "Put mug on coffeetable":        [("mug", "on", "coffeetable")],
    "Put cutleryfork on coffeetable":[("cutleryfork", "on", "coffeetable")],
    "Put coffeepot on coffeetable":  [("coffeepot", "on", "coffeetable")],
    "Put waterglass on coffeetable": [("waterglass", "on", "coffeetable")],
    "Put plate on microwave":        [("plate", "on", "microwave")],
    "Put salmon on microwave":       [("salmon", "on", "microwave")],
    "Put bananas on microwave":      [("bananas", "on", "microwave")],
    "Put apple on microwave":        [("apple", "on", "microwave")],
    "Put mug on microwave":          [("mug", "on", "microwave")],
    "Put mug to tvstand":            [("mug", "on", "tvstand")],
    "Put book on tvstand":           [("book", "on", "tvstand")],
    "Put notes to tvstand":          [("notes", "on", "tvstand")],
    "Put waterglass on tvstand":     [("waterglass", "on", "tvstand")],
    "Put remotecontrol on tvstand":  [("remotecontrol", "on", "tvstand")],
    "Put cellphone on bed":          [("cellphone", "on", "bed")],
    "Put keyboard on bed":           [("keyboard", "on", "bed")],
    "Put folder on bed":             [("folder", "on", "bed")],
    "Put cereal on bed":             [("cereal", "on", "bed")],
    "Put box on bed":                [("box", "on", "bed")],

    "Place towel in closet":         [("towel", "inside", "closet")],
    "Place keyboard in closet":      [("keyboard", "inside", "closet")],
    "Place folder in closet":        [("folder", "inside", "closet")],
    "Place mouse in closet":         [("mouse", "inside", "closet")],
    "Place book in bookshelf":       [("book", "inside", "bookshelf")],
    "Place folder in bookshelf":     [("folder", "inside", "bookshelf")],
    "Place magazine in bookshelf":   [("magazine", "inside", "bookshelf")],
    "Place paper in bookshelf":      [("paper", "inside", "bookshelf")],
    "Place box in bookshelf":        [("box", "inside", "bookshelf")],
    "Place plum in cabinet":         [("plum", "inside", "cabinet")],
    "Place mug in cabinet":          [("mug", "inside", "cabinet")],
    "Place cereal in cabinet":       [("cereal", "inside", "cabinet")],
    "Place cupcake in cabinet":      [("cupcake", "inside", "cabinet")],
    "Place paper in cabinet":        [("paper", "inside", "cabinet")],
    "Place towel in cabinet":        [("towel", "inside", "cabinet")],
    "Place box in cabinet":          [("box", "inside", "cabinet")],
    "Place keyboard in cabinet":     [("keyboard", "inside", "cabinet")],
    "Place box in closet":           [("box", "inside", "closet")],
    "Place slippers in closet":      [("slippers", "inside", "closet")],
    "Place cutleryknife in dishwasher":[("cutleryknife", "inside", "dishwasher")],
    "Place plate in dishwasher":     [("plate", "inside", "dishwasher")],
    "Place mug in dishwasher":       [("mug", "inside", "dishwasher")],
    "Place cutleryfork in dishwasher":[("cutleryfork", "inside", "dishwasher")],
    "Place milk in fridge":          [("milk", "inside", "fridge")],
    "Place juice in fridge":         [("juice", "inside", "fridge")],
    "Place apple in fridge":         [("apple", "inside", "fridge")],
    "Place plum in fridge":          [("plum", "inside", "fridge")],
    "Place carrot in fridge":        [("carrot", "inside", "fridge")],
    "Place breadslice in microwave": [("breadslice", "inside", "microwave")],
    "Place cupcake in microwave":    [("cupcake", "inside", "microwave")],
    "Place pancake in microwave":    [("pancake", "inside", "microwave")],
    "Place pie in microwave":        [("pie", "inside", "microwave")],
}
# fmt: on

TASKS: list[str] = list(TASKS_SET.keys())
assert len(TASKS) == 78, f"expected 78 tasks, got {len(TASKS)}"

# Stratified-by-index seen-task split. 2 turnon + 2 open + 6 puton + 6 placein.
SEEN_TASKS: list[int] = [0, 4, 9, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74]
assert len(SEEN_TASKS) == 16

UNSEEN_TASKS: list[int] = [i for i in range(len(TASKS)) if i not in SEEN_TASKS]
assert len(UNSEEN_TASKS) == 62
UNSEEN_TASKS_HALF: list[int] = UNSEEN_TASKS[-24:]

# TMoW house IDs (not directly used by WorMI build because raw VH file IDs
# differ; kept for reference). WorMI uses local init-graph cache indices.
SEEN_DOMAIN_TMOW: list[int]   = [18, 20, 22, 24, 26, 28, 29, 31, 32, 34]
UNSEEN_DOMAIN_TMOW: list[int] = [0, 1, 5, 6, 7, 8, 9, 12, 13, 15]

ROOMS: list[str] = ["kitchen", "bedroom", "livingroom", "bathroom"]


def task_family(task_idx: int) -> str:
    """Return WorMI family name for a TMoW task index."""
    if task_idx < 9:
        return "turnon"
    if task_idx < 16:
        return "open"
    if task_idx < 46:
        return "puton"
    return "placein"


_PUTON_FIXUP = {"to": "on"}  # TMoW has "Put mug to tvstand"; treat "to" == "on".


def task_args(task_idx: int) -> tuple[str, ...]:
    """Parse a TMoW task name back to WorMI (family, args).

    "Turn on tv"             -> ("tv",)
    "Open cabinet"           -> ("cabinet",)
    "Put apple on desk"      -> ("apple", "desk")
    "Place mouse in closet"  -> ("mouse", "closet")
    """
    name = TASKS[task_idx]
    parts = name.lower().split()
    family = task_family(task_idx)
    if family == "turnon":
        return (parts[2],)
    if family == "open":
        return (parts[1],)
    if family == "puton":
        # Put <obj> on/to <target>
        return (parts[1], parts[3])
    if family == "placein":
        # Place <obj> in <target>
        return (parts[1], parts[3])
    raise ValueError(family)


def task_to_tuple(task_idx: int) -> tuple[str, tuple[str, ...]]:
    """Return (family, args) tuple."""
    return task_family(task_idx), task_args(task_idx)


def goal_triple(task_idx: int) -> tuple[str, str, str]:
    """Return the canonical TMoW goal triple."""
    return TASKS_SET[TASKS[task_idx]][0]
