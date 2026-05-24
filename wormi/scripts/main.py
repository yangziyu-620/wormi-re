import logging
import sys

import wormi.scripts.eval
import wormi.scripts.eval_table1
import wormi.scripts.eval_vh_rollout
import wormi.scripts.eval_world
import wormi.scripts.train
import wormi.scripts.train_world

_AVAILABLE_SCRIPTS = {
    "train": wormi.scripts.train.main,
    "eval": wormi.scripts.eval.main,
    "eval-table1": wormi.scripts.eval_table1.main,
    "eval-vh-rollout": wormi.scripts.eval_vh_rollout.main,
    "world": {
        "train": wormi.scripts.train_world.main,
        "eval": wormi.scripts.eval_world.main,
    },
}

logger = logging.getLogger(__name__)


def help(cmds, target, level=0):
    def _print(*s: str, **kwargs):
        print("    " * level + " ".join(s), **kwargs)

    print("Usage: wormi", *cmds, "<subcommand> [options]")
    _print("Subcommands:")
    for name, func in target.items():
        if isinstance(func, dict):
            _print(f"  {name}: ", end="")
            help([*cmds, name], func, level + 1)
        _print(f"  {name}: {func.__doc__}")
    sys.exit(0)


def main():
    if len(sys.argv) < 2:
        logger.error(
            f"wormi: missing subcommand. Use one of the following subcommands: "
            f"{', '.join(_AVAILABLE_SCRIPTS.keys())}\n"
            f"Try 'wormi --help' for more information."
        )
        sys.exit(1)

    args = sys.argv[1:]
    if args[0] in ["help", "-h", "--help"]:
        print(
            "wormi: a framework for training and evaluating models on multimodal tasks."
        )
        help([], _AVAILABLE_SCRIPTS)

    target = _AVAILABLE_SCRIPTS
    cmds = list[str]()
    while isinstance(target, dict):
        subcommand = args.pop(0)
        if subcommand in ["help", "-h", "--help"]:
            help(cmds, target)
        if subcommand not in target:
            logger.error(
                f"wormi: invalid subcommand '{subcommand}'. Use one of the following "
                f"subcommands: {', '.join(target.keys())}\n"
                f"Try 'wormi {' '.join([*cmds, '--help'])}' for more information."
            )
            sys.exit(1)
        target = target[subcommand]
        cmds.append(subcommand)
    sys.argv[0] = "wormi " + " ".join(cmds)
    target(args)
