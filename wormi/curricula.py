from __future__ import annotations

import ast
import logging
import re
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

from wormi.model import WorMIIntegrateMethod

if TYPE_CHECKING:
    from wormi.trainer import WorMITrainerConfig

logger = logging.getLogger(__name__)


@dataclass
class WorldModel:
    model_name: str
    connections: list[int] | None = None

    def __init__(
        self,
        model_name: str,
        connections: list[int] | None = None,
    ):
        self.model_name = model_name
        self.connections = connections


@dataclass
class WorldModelCurriculum:
    base_model: str
    dataset: Path
    tokenizer: str
    name: str | None
    num_train_samples: int | None
    num_eval_samples: int | None
    trainer_args: WorMITrainerConfig
    behavior_cloning: bool = False

    def __init__(
        self,
        base_model: str,
        dataset: Path | str,
        tokenizer: str | None = None,
        name: str | None = None,
        num_train_samples: int | None = None,
        num_eval_samples: int | None = None,
        trainer_args: WorMITrainerConfig | None = None,
        behavior_cloning: bool = False,
    ):
        self.base_model = base_model
        self.tokenizer = tokenizer or base_model
        self.dataset = Path(dataset)
        self.name = name
        self.num_train_samples = num_train_samples
        self.num_eval_samples = num_eval_samples
        self.trainer_args = deepcopy(trainer_args) or WorMITrainerConfig()
        self.behavior_cloning = behavior_cloning


@dataclass
class WorldModelCurricula:
    output_dir: Path | None
    curricula: list[WorldModelCurriculum]

    def __init__(
        self,
        curricula: list[WorldModelCurriculum],
        output_dir: Path | None = None,
    ):
        self.curricula = deepcopy(curricula)
        self.output_dir = Path(output_dir or "/tmp/hf")


@dataclass
class WorMICurriculum:
    world_models: list[int]
    datasets: list[int]
    name: str | None
    num_train_samples: int | None
    num_eval_samples: int | None
    trainer_args: WorMITrainerConfig

    def __init__(
        self,
        target_world_models: list[int],
        datasets: list[int],
        name: str | None = None,
        num_train_samples: int | None = None,
        num_eval_samples: int | None = None,
        trainer_args: WorMITrainerConfig | None = None,
    ):
        self.world_models = target_world_models
        self.datasets = datasets
        self.name = name
        self.num_train_samples = num_train_samples
        self.num_eval_samples = num_eval_samples
        self.trainer_args = deepcopy(trainer_args) or WorMITrainerConfig()


@dataclass
class WorMICurricula:
    name: str
    base_model: str
    world_models: list[WorldModel]
    datasets: list[Path]
    train: list[WorMICurriculum]
    test: list[WorMICurriculum]
    connections: list[int]
    method: WorMIIntegrateMethod
    num_heads: int
    self_attention: bool
    model_wise_positional_encoding: bool
    resume_from: str | None
    meta_learning: bool
    meta_learning_rate: float
    num_iterations: int
    decaying_learning_rate: bool
    output_dir: Path
    run_name: str
    test_continuously: bool
    vision: bool
    # Paper Algorithm 1 lines 18-31: Wasserstein-prototype top-K retrieval of
    # world models at test time. Set sentence_embedding_model to enable; if
    # None, eval falls back to the fixed `curriculum.world_models` indices.
    sentence_embedding_model: str | None
    retrieval_k: int
    prototype_size: int

    def __init__(
        self,
        base_model: str,
        world_models: list[WorldModel],
        datasets: list[str],
        train_curricula: list[WorMICurriculum],
        test_curricula: list[WorMICurriculum],
        connections: list[int] = [7, 15],
        method: WorMIIntegrateMethod = WorMIIntegrateMethod.CONCAT,
        num_heads: int = 8,
        self_attention: bool = True,
        model_wise_positional_encoding: bool = True,
        resume_from: str | None = None,
        meta_learning: bool = False,
        meta_learning_rate: float = 1.0,
        num_iterations: int = 1,
        decaying_learning_rate: bool = False,
        name: str | None = None,
        output_dir: Path | str | None = None,
        run_name: str | None = None,
        test_continuously: bool = False,
        vision: bool | None = None,
        sentence_embedding_model: str | None = None,
        retrieval_k: int = 3,
        prototype_size: int = 15,
    ):
        self.name = name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.base_model = base_model
        self.connections = deepcopy(connections)
        self.method = method
        self.num_heads = num_heads
        self.self_attention = self_attention
        self.model_wise_positional_encoding = model_wise_positional_encoding
        self.world_models = deepcopy(world_models)
        self.datasets = list(map(Path, datasets))
        self.train = deepcopy(train_curricula)
        self.test = deepcopy(test_curricula)
        self.resume_from = resume_from
        self.meta_learning = meta_learning
        self.meta_learning_rate = meta_learning_rate
        self.num_iterations = num_iterations
        self.decaying_learning_rate = decaying_learning_rate
        self.output_dir = Path(output_dir or "/tmp/hf")
        self.run_name = run_name or self.name
        self.test_continuously = test_continuously

        if vision is None and "vision" in self.base_model.lower():
            logger.warning(
                "Vision model detected. Setting vision to True. If this is "
                "incorrect, please set the vision parameter."
            )
            self.vision = True
        else:
            self.vision = vision or False

        self.sentence_embedding_model = sentence_embedding_model
        self.retrieval_k = retrieval_k
        self.prototype_size = prototype_size

        for curr in self.train:
            curr.trainer_args.output_dir = self.output_dir
            curr.trainer_args.run_name = self.run_name

        for curr in self.test:
            curr.trainer_args.output_dir = None

    def merge(self, other: WorMICurricula):
        assert self.base_model == other.base_model, "Main models must match"
        assert self.connections == other.connections, "Connections must match"
        return WorMICurricula(
            name=f"{self.name}+{other.name}",
            base_model=self.base_model,
            connections=self.connections,
            world_models=self.world_models + other.world_models,
            datasets=list(map(str, self.datasets + other.datasets)),
            train_curricula=self.train + other.train,
            test_curricula=self.test + other.test,
            method=self.method,
            num_heads=self.num_heads,
            self_attention=self.self_attention,
            model_wise_positional_encoding=self.model_wise_positional_encoding,
            resume_from=self.resume_from,
            meta_learning=self.meta_learning,
            meta_learning_rate=self.meta_learning_rate,
            num_iterations=self.num_iterations,
            decaying_learning_rate=self.decaying_learning_rate,
            output_dir=self.output_dir,
            run_name=self.run_name,
            test_continuously=self.test_continuously,
            vision=self.vision,
            sentence_embedding_model=self.sentence_embedding_model,
            retrieval_k=self.retrieval_k,
            prototype_size=self.prototype_size,
        )


def load_world_model_curricula(path: str | Path) -> WorldModelCurricula:
    path = Path(path)
    with open(path, "r") as f:
        node = ast.parse(f.read())
    obj = compile(node, "<ast>", "exec")
    loc = {}
    exec(obj, globals(), loc)
    curricula = cast(WorldModelCurricula, loc["curricula"])

    if curricula.output_dir == Path("/tmp/hf"):
        curricula.output_dir = path.parent

    return curricula


def load_wormi_curricula(path: str | Path) -> WorMICurricula:
    path = Path(path)
    with open(path, "r") as f:
        node = ast.parse(f.read())
    obj = compile(node, "<ast>", "exec")
    loc = {}
    exec(obj, globals(), loc)
    curricula = cast(WorMICurricula, loc["curricula"])

    if re.match(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", curricula.name):
        if curricula.run_name == curricula.name:
            curricula.run_name = path.parent.name
            for curr in curricula.train:
                curr.trainer_args.run_name = path.parent.name
        curricula.name = path.parent.name

    if curricula.output_dir == Path("/tmp/hf"):
        curricula.output_dir = path.parent.parent
        for curr in curricula.train:
            curr.trainer_args.output_dir = curricula.output_dir

    return curricula
