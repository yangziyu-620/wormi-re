from __future__ import annotations

import logging
import re
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import reduce
from pathlib import Path
from threading import Lock, Thread
from typing import Callable, Generic, TypeVar, override

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    EvalPrediction,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers import SchedulerType as _SchedulerType
from transformers import TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

from wormi.curricula import WorMICurricula, WorMICurriculum
from wormi.datasets.auto_jsonl import AutoJsonlDataset
from wormi.datasets.jsonl import JsonlDataset
from wormi.model import WorMI

logger = logging.getLogger(__name__)


class SchedulerType(Enum):
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"
    INVERSE_SQRT = "inverse_sqrt"
    REDUCE_ON_PLATEAU = "reduce_lr_on_plateau"
    COSINE_WITH_MIN_LR = "cosine_with_min_lr"
    WARMUP_STABLE_DECAY = "warmup_stable_decay"


@dataclass
class WorMITrainerConfig:
    max_steps: int = field(
        default=1000, metadata={"help": "Maximum number of training steps."}
    )
    eval_steps: int = field(
        default=500, metadata={"help": "Number of steps between evaluations."}
    )
    logging_steps: int = field(
        default=100, metadata={"help": "Number of steps between logging."}
    )
    save_steps: int = field(
        default=100, metadata={"help": "Number of steps between saving."}
    )
    batch_size: int = field(
        default=1, metadata={"help": "Batch size for training."}
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of update steps to accumulate before optimizer step."},
    )
    learning_rate: float = field(
        default=5e-5, metadata={"help": "Learning rate for training."}
    )
    lr_scheduler_type: SchedulerType = field(
        default=SchedulerType.LINEAR,
        metadata={"help": "Type of learning rate scheduler."},
    )
    output_dir: Path | None = field(
        default=None, metadata={"help": "Output directory for saving models."}
    )
    run_name: str | None = field(
        default=None, metadata={"help": "Name of the run."}
    )
    logging_dir: Path | str | None = field(
        default=None, metadata={"help": "Directory for saving logs."}
    )

    def copy_with(
        self,
        max_steps: int | None = None,
        eval_steps: int | None = None,
        logging_steps: int | None = None,
        save_steps: int | None = None,
        batch_size: int | None = None,
        gradient_accumulation_steps: int | None = None,
        learning_rate: float | None = None,
        lr_scheduler_type: SchedulerType | None = None,
        output_dir: Path | None = None,
        run_name: str | None = None,
        logging_dir: Path | str | None = None,
    ) -> WorMITrainerConfig:
        return WorMITrainerConfig(
            max_steps=max_steps if max_steps is not None else self.max_steps,
            eval_steps=(
                eval_steps if eval_steps is not None else self.eval_steps
            ),
            logging_steps=(
                logging_steps
                if logging_steps is not None
                else self.logging_steps
            ),
            save_steps=(
                save_steps if save_steps is not None else self.save_steps
            ),
            batch_size=(
                batch_size if batch_size is not None else self.batch_size
            ),
            gradient_accumulation_steps=(
                gradient_accumulation_steps
                if gradient_accumulation_steps is not None
                else self.gradient_accumulation_steps
            ),
            learning_rate=(
                learning_rate
                if learning_rate is not None
                else self.learning_rate
            ),
            lr_scheduler_type=(
                lr_scheduler_type
                if lr_scheduler_type is not None
                else self.lr_scheduler_type
            ),
            output_dir=(
                output_dir if output_dir is not None else self.output_dir
            ),
            run_name=run_name if run_name is not None else self.run_name,
            logging_dir=(
                logging_dir if logging_dir is not None else self.logging_dir
            ),
        )


class WorMISubTrainer(SFTTrainer):
    def __init__(
        self,
        model: PreTrainedModel,
        args: WorMITrainerConfig,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        tokenizer: PreTrainedTokenizerBase,
        compute_metrics: Callable[[EvalPrediction], dict] | None = None,
        callbacks: list[TrainerCallback] | None = None,
    ):
        if args.output_dir is not None:
            output_dir = args.output_dir
        else:
            output_dir = Path("/tmp/hf")

        if args.run_name is not None:
            self.__dirname = args.run_name
        else:
            trials = {0}
            date = datetime.now().strftime("%Y-%m-%d")
            if output_dir.exists():
                for dirname in output_dir.iterdir():
                    if not dirname.is_dir():
                        continue
                    if m := re.match(
                        r"(\d{4}-\d{2}-\d{2})_(\d{3})", dirname.name
                    ):
                        cur_date, cur_trial = m.groups()
                        if cur_date == date:
                            trials.add(int(cur_trial))
            next_trial = max(trials) + 1
            self.__dirname = (
                f"{datetime.now().strftime('%Y-%m-%d')}_{next_trial:03d}"
            )

        self.__output_dir = output_dir / self.__dirname

        if args.logging_steps == 0:
            logging_strategy = "no"
        else:
            logging_strategy = "steps"

        if args.save_steps == 0:
            save_strategy = "no"
        else:
            save_strategy = "steps"

        if args.logging_dir is not None:
            if isinstance(args.logging_dir, Path):
                logging_dir = args.logging_dir
            else:
                logging_dir = Path(
                    self.__output_dir / "logs" / args.logging_dir
                )
        else:
            logging_dir = self.__output_dir / "logs"

        lr_scheduler_type = _SchedulerType(args.lr_scheduler_type.value)
        optimizer = None

        config = SFTConfig(
            output_dir=str(self.__output_dir),
            overwrite_output_dir=True,
            logging_dir=str(logging_dir),
            do_train=True,
            do_eval=True,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            eval_strategy="steps",
            logging_strategy=logging_strategy,
            save_strategy=save_strategy,
            max_steps=args.max_steps,
            eval_steps=args.eval_steps,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            learning_rate=args.learning_rate,
            lr_scheduler_type=lr_scheduler_type,
            label_names=[],
            report_to=["tensorboard"],
            dataset_text_field="text",
            # VH paper-style triple-list observations + Llama-3 chat envelope
            # push every sample past trl 0.11's default 1024-token cutoff,
            # truncating the response template and zeroing out the loss mask.
            max_seq_length=4096,
            # batch=4 × seq=4096 × Llama-3.2-1B fp32 + AdamW OOMs an L40S
            # (44.6 GiB); bf16 + activation checkpointing fits inside ~25 GiB.
            bf16=True,
            # WorMI.gradient_checkpointing_enable() raises ValueError —
            # the composed base+world hook graph isn't checkpointable.
            gradient_checkpointing=False,
            # CLI does not expose resume_from_checkpoint, so the optimizer +
            # scheduler + rng state that the Trainer writes alongside each
            # checkpoint-N/ dir is dead weight (~10 GiB extra per ckpt).
            # save_only_model keeps just the model+config+tokenizer files.
            save_only_model=True,
        )

        self.__tokenizer = tokenizer
        self.__model = model
        self.__args = config
        self.__train_dataset = train_dataset
        self.__eval_dataset = eval_dataset

        self.__tokenizer.pad_token = "<|end_of_text|>"

        data_collator = DataCollatorForCompletionOnlyLM(
            response_template="<|start_header_id|>assistant<|end_header_id|>",
            tokenizer=self.__tokenizer,
        )

        super().__init__(
            model=self.__model,
            args=self.__args,
            train_dataset=self.__train_dataset,
            eval_dataset=self.__eval_dataset,
            tokenizer=self.__tokenizer,
            data_collator=data_collator,
            optimizers=(optimizer, None),  # type: ignore
            compute_metrics=compute_metrics,
            callbacks=callbacks,
        )
        self.can_return_loss = True

    def train(  # type: ignore
        self,
        save_model: bool = True,
        model_name: str = "last",
        resume_from_checkpoint: bool = False,
    ):
        super().train(resume_from_checkpoint=resume_from_checkpoint)  # type: ignore
        if save_model:
            self.save_model(str(self.__output_dir / model_name))

    def save_model(self, output_dir: str, **kwargs):  # type: ignore
        if kwargs.pop("relative", False):
            output_dir = str(self.__output_dir / output_dir)
        self.__model.save_pretrained(output_dir)
        self.__model.config.save_pretrained(output_dir)
        self.__tokenizer.save_pretrained(output_dir)

    def test(
        self,
        test_dataset: Dataset,
        interactive: bool = True,
        sample: int = -1,
        use_cache: bool = True,
    ):
        if sample < 0:
            dataset = test_dataset
        else:
            dataset = test_dataset.shuffle(seed=42).select(range(sample))
        for elem in dataset:
            prompt = elem["text"]  # type: ignore
            prompt = "assistant".join(prompt.split("assistant")[:-1])
            prompt += "assistant<|end_header_id|>\n\n"

            input_ids = self.__tokenizer(prompt, return_tensors="pt")
            input_ids = input_ids.to(self.__model.device)  # type: ignore

            pred = self.__model.generate(
                **input_ids,  # type: ignore
                max_length=4096,
                pad_token_id=self.__tokenizer.pad_token_id,
                eos_token_id=self.__tokenizer.eos_token_id,
                tokenizer=self.__tokenizer,
                use_cache=use_cache,
            )[0]
            pred_decoded = (
                self.__tokenizer.decode(pred, skip_special_tokens=False)
                .split("assistant<|end_header_id|>")[-1]
                .split("<|eot_id|>")[0]
                .strip()
            )
            print("Predicted:", pred_decoded)

            if interactive:
                cmd = input()
                if cmd == "quit":
                    break


class WorMITrainer:
    def __init__(
        self,
        model: WorMI,
        tokenizer: PreTrainedTokenizerBase,
        curricula: WorMICurricula,
        synchronize_trainers: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.curricula = curricula
        self.step = 0
        self.current_trainer: WorMISubTrainer | None = None
        self.synchronize_trainers = synchronize_trainers
        if self.synchronize_trainers:
            self.max_steps = (
                sum((x.trainer_args.max_steps for x in curricula.train))
                * curricula.num_iterations
            )
        else:
            self.max_steps = [
                x.trainer_args.max_steps * curricula.num_iterations
                for x in curricula.train
            ]

    def trainer_callback_factory(
        self,
        i: int,
        curriculum: WorMICurriculum,
        synchronize_trainers: bool = True,
    ):
        return WorMITrainerCallback(
            self, i, curriculum, synchronize_trainers=synchronize_trainers
        )

    def train(self, **options):
        self.iter = 0
        self.step = 0
        self.start_curriculum_step = 0
        self.ready = False
        self.training_lock = [Lock() for _ in range(len(self.curricula.train))]
        self.trainers: list[WorMISubTrainer] = []
        self.trainer_threads: list[Thread] = []
        self.panic: RuntimeError | None = None
        self.device = self.model.device

        if not self.model.has_model_been_built:
            model = self.curricula.world_models[
                self.curricula.train[0].world_models[0]
            ]
            self.model.implant(
                AutoModelForCausalLM.from_pretrained(
                    model.model_name,
                    torch_dtype=torch.bfloat16,
                ),
                model.connections,
            )

        def thread_target(i: int, curriculum: WorMICurriculum):
            self.training_lock[i].acquire()

            print(f"\r🏃‍➡️ Trainer {i + 1} started", flush=True)

            target_datasets = [
                self.curricula.datasets[j] for j in curriculum.datasets
            ]

            train_datasets = [
                AutoJsonlDataset.load(x / "train.jsonl", **options)
                for x in target_datasets
            ]
            test_datasets = [
                AutoJsonlDataset.load(x / "test.jsonl", **options)
                for x in target_datasets
            ]

            train_dataset = reduce(JsonlDataset.merge, train_datasets)
            test_dataset = reduce(JsonlDataset.merge, test_datasets)

            train_dataset = train_dataset.as_chat(self.tokenizer).shuffle()
            test_dataset = test_dataset.as_chat(self.tokenizer).shuffle()

            if curriculum.num_train_samples is not None:
                if curriculum.num_train_samples > len(train_dataset):
                    raise ValueError(
                        f"Number of train samples ({curriculum.num_train_samples}) "
                        f"exceeds the total number of samples ({len(train_dataset)})."
                    )
                train_dataset = train_dataset.take(curriculum.num_train_samples)

            if curriculum.num_eval_samples is not None:
                if curriculum.num_eval_samples > len(test_dataset):
                    raise ValueError(
                        f"Number of eval samples ({curriculum.num_eval_samples}) "
                        f"exceeds the total number of samples ({len(test_dataset)})."
                    )
                test_dataset = test_dataset.take(curriculum.num_eval_samples)

            trainer_args = deepcopy(curriculum.trainer_args)
            if isinstance(self.max_steps, list):
                trainer_args.max_steps = self.max_steps[i]
            else:
                trainer_args.max_steps = self.max_steps
            if curriculum.name is not None and trainer_args.logging_dir is None:
                trainer_args.logging_dir = curriculum.name
            trainer = WorMISubTrainer(
                model=self.model,
                args=trainer_args,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
                tokenizer=self.tokenizer,
            )
            trainer.add_callback(
                self.trainer_callback_factory(
                    i,
                    curriculum,
                    synchronize_trainers=self.synchronize_trainers,
                )
            )
            self.trainers.append(trainer)
            try:
                trainer.train(save_model=(i == len(self.curricula.train) - 1))
            except RuntimeError as e:
                self.panic = e

        for lock in self.training_lock:
            lock.acquire(blocking=False)

        self.trainer_threads = [
            Thread(target=thread_target, args=(i, curriculum))
            for i, curriculum in enumerate(self.curricula.train)
        ]
        for thread in self.trainer_threads:
            thread.start()

        self.training_lock[0].release()

        while any(thread.is_alive() for thread in self.trainer_threads):
            for thread in self.trainer_threads:
                if thread.is_alive():
                    thread.join(timeout=1)
                    if self.panic:
                        raise RuntimeError("Training failed") from self.panic

    def _inner_test_loop(
        self,
        num_iter: int,
        curriculum: WorMICurriculum,
        /,
        num_samples: int = 5,
        interactive: bool = True,
        **options,
    ):
        total = len(self.curricula.test) * self.curricula.num_iterations
        print(f"Testing curriculum {num_iter + 1}/{total}")

        target_datasets = [
            self.curricula.datasets[i] for i in curriculum.datasets
        ]
        test_datasets = [
            AutoJsonlDataset.load(x / "test.jsonl", **options)
            for x in target_datasets
        ]
        test_dataset = reduce(JsonlDataset.merge, test_datasets)
        test_dataset = test_dataset.as_chat(self.tokenizer).shuffle()

        if curriculum.num_eval_samples is not None:
            test_dataset = test_dataset.take(curriculum.num_eval_samples)

        target_models = [
            self.curricula.world_models[i] for i in curriculum.world_models
        ]
        self.model.remove_all()
        for model in target_models:
            aux_model = AutoModelForCausalLM.from_pretrained(
                model.model_name, torch_dtype=torch.bfloat16
            )
            self.model.implant(aux_model, model.connections)
        self.model.to(self.device)

        trainer = WorMISubTrainer(
            model=self.model,
            args=curriculum.trainer_args,
            train_dataset=test_dataset,
            eval_dataset=test_dataset,
            tokenizer=self.tokenizer,
        )
        trainer.test(
            test_dataset,
            interactive=interactive,
            sample=num_samples if not interactive else -1,
            use_cache=False,
        )

    def test(self, num_samples: int = 5, interactive: bool = True, **options):
        for i, curriculum in enumerate(self.curricula.test):
            self._inner_test_loop(
                i, curriculum, num_samples, interactive, **options
            )


T = TypeVar("T", bound=WorMITrainer)


class WorMITrainerCallback(Generic[T], TrainerCallback):
    def __init__(
        self,
        main_trainer: T,
        trainer_idx: int,
        curriculum: WorMICurriculum,
        synchronize_trainers: bool = True,
    ):
        self.main_trainer = main_trainer
        self.trainer_idx = trainer_idx
        self.curriculum = curriculum
        self.synchronize_trainers = synchronize_trainers

    @property
    def next_trainer_idx(self):
        num_trainers = len(self.curricula.train)
        next_trainer_idx = (self.trainer_idx + 1) % num_trainers
        return next_trainer_idx

    @property
    def current_trainer(self):
        return self.main_trainer.trainers[self.trainer_idx]

    @property
    def lock(self):
        return self.main_trainer.training_lock[self.trainer_idx]

    @property
    def next_lock(self):
        return self.main_trainer.training_lock[self.next_trainer_idx]

    @property
    def global_iter(self):
        return self.main_trainer.iter

    @global_iter.setter
    def global_iter(self, value: int):
        self.main_trainer.iter = value

    @property
    def global_step(self):
        return self.main_trainer.step

    @global_step.setter
    def global_step(self, value: int):
        self.main_trainer.step = value

    @property
    def curricula(self):
        return self.main_trainer.curricula

    @property
    def model(self):
        return self.main_trainer.model

    @property
    def device(self):
        return self.main_trainer.device

    @override
    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if self.synchronize_trainers and self.trainer_idx > 0:
            main_trainer = self.main_trainer.trainers[0]
            self.current_trainer.callback_handler.lr_scheduler = (
                self.current_trainer.lr_scheduler
            ) = main_trainer.lr_scheduler

        super().on_train_begin(args, state, control, **kwargs)

        print(f"\r🏃‍➡️ Trainer {self.trainer_idx + 1} ready", flush=True)
        if self.next_trainer_idx == 0:
            self.main_trainer.ready = True
            print("\r🚦 All trainers ready", flush=True)

    @override
    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if self.lock.locked():
            self.next_lock.release()
        self.lock.acquire()

        if self.synchronize_trainers:
            state.global_step = self.global_step

        if self.global_step == self.main_trainer.start_curriculum_step:
            if self.trainer_idx == 0:
                self.on_iteration_start(args, state, control, **kwargs)
            self.on_curriculum_start(args, state, control, **kwargs)

            self.model.remove_all()
            for target in self.curriculum.world_models:
                model = self.curricula.world_models[target]
                aux_model = AutoModelForCausalLM.from_pretrained(
                    model.model_name, torch_dtype=torch.bfloat16
                )
                aux_model.to(self.device)
                self.model.implant(aux_model, model.connections)
            self.model.to(self.device)

        super().on_step_begin(args, state, control, **kwargs)

    @override
    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        super().on_step_end(args, state, control, **kwargs)

        if self.synchronize_trainers:
            for trainer in self.main_trainer.trainers:
                trainer._globalstep_last_logged = self.global_step

        self.global_step += 1

        if (
            self.global_step - self.main_trainer.start_curriculum_step
            >= self.curriculum.trainer_args.max_steps
        ):
            self.main_trainer.start_curriculum_step = self.global_step

            self.on_curriculum_end(args, state, control, **kwargs)
            if self.global_iter == self.curricula.num_iterations - 1:
                control.should_training_stop = True
            if self.next_trainer_idx == 0:
                self.global_iter += 1
                self.on_iteration_end(args, state, control, **kwargs)
        else:
            self.lock.release()

    @override
    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self.next_lock.release()
        super().on_train_end(args, state, control, **kwargs)

    def on_iteration_start(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        pass

    def on_iteration_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        pass

    def on_curriculum_start(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        pass

    def on_curriculum_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        pass


class WorMIMetaLearningTrainer(WorMITrainer):
    def __init__(
        self,
        model: WorMI,
        tokenizer: PreTrainedTokenizerBase,
        curricula: WorMICurricula,
    ):
        super().__init__(
            model, tokenizer, curricula, synchronize_trainers=False
        )

        self.save_steps = 0
        for curriculum in curricula.train:
            if curriculum.trainer_args.save_steps != 0:
                if self.save_steps == 0:
                    self.save_steps = curriculum.trainer_args.save_steps
                elif self.save_steps != curriculum.trainer_args.save_steps:
                    logger.warning(
                        "Different save_steps values in curricula, using the first one"
                    )
                curriculum.trainer_args.save_steps = 0

    @override
    def trainer_callback_factory(
        self,
        i: int,
        curriculum: WorMICurriculum,
        *args,
        **kwargs,
    ):
        return MetaLearningAggregationCallback(self, i, curriculum)

    @override
    def train(self, **options):
        self.all_params = [
            list(p.detach().cpu() for p in self.model.parameters())
            for _ in range(len(self.curricula.train))
        ]
        self.iteration_start_params = [
            p.detach().cpu().clone() for p in self.model.parameters()
        ]
        super().train(**options)


class MetaLearningAggregationCallback(
    WorMITrainerCallback[WorMIMetaLearningTrainer]
):
    def __init__(
        self,
        main_trainer: WorMIMetaLearningTrainer,
        trainer_idx: int,
        curriculum: WorMICurriculum,
    ):
        super().__init__(
            main_trainer, trainer_idx, curriculum, synchronize_trainers=False
        )
        self.local_step = 0

    @property
    def save_steps(self):
        return self.main_trainer.save_steps

    @property
    def params(self):
        return self.main_trainer.all_params[self.trainer_idx]

    @property
    def meta_learning_rate(self):
        return self.curricula.meta_learning_rate

    @override
    def on_iteration_start(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self.main_trainer.iteration_start_params = [
            p.detach().cpu().clone() for p in self.params
        ]

    @override
    def on_iteration_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        super().on_iteration_end(args, state, control, **kwargs)

        for params in self.main_trainer.all_params[1:]:
            for main_param, p in zip(self.main_trainer.all_params[0], params):
                main_param.data = main_param.data.cuda() + p.data.cuda()

        for old_param, main_param in zip(
            self.main_trainer.iteration_start_params,
            self.main_trainer.all_params[0],
        ):
            mean_param = main_param.data / len(self.main_trainer.all_params)
            old_param_data = old_param.data.to(mean_param.device)
            main_param.data = (
                old_param_data
                + self.meta_learning_rate
                * (mean_param - old_param_data)
            ).cpu()

        for params in self.main_trainer.all_params[1:]:
            for main_param, p in zip(self.main_trainer.all_params[0], params):
                p.data = main_param.data.cpu()

    @override
    def on_curriculum_start(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        for model_param, params in zip(self.model.parameters(), self.params):
            model_param.data.copy_(params.data)
            model_param.data.to(self.device)
            model_param.requires_grad = True

        super().on_curriculum_start(args, state, control, **kwargs)

    @override
    def on_curriculum_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        super().on_curriculum_end(args, state, control, **kwargs)

        for model_param, param in zip(self.model.parameters(), self.params):
            param.data = model_param.data.detach().cpu()
            model_param.data.to(self.device)

    @override
    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        super().on_step_end(args, state, control, **kwargs)
        self.local_step += 1

        if (
            self.save_steps != 0
            and self.trainer_idx == len(self.main_trainer.trainers) - 1
            and self.local_step % self.save_steps == 0
        ):
            self.current_trainer.save_model(
                f"checkpoint-{self.local_step}", relative=True
            )
            print(f"\r💾 Model saved at step {self.local_step}", flush=True)
