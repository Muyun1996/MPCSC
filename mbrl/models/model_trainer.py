# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import copy
import functools
import itertools
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import tqdm
from torch import optim as optim

from mbrl.util.logger import Logger
from mbrl.util.replay_buffer import BootstrapIterator, TransitionIterator

from .model import Model

import matplotlib.pyplot as plt
import wandb

MODEL_LOG_FORMAT = [
    ("train_iteration", "I", "int"),
    ("epoch", "E", "int"),
    ("train_dataset_size", "TD", "int"),
    ("val_dataset_size", "VD", "int"),
    ("test_dataset_size", "testD", "int"),
    ("model_loss", "MLOSS", "float"),
    ("model_val_score", "MVSCORE", "float"),
    ("model_best_val_score", "MBVSCORE", "float"),
    ("model_test_score", "MTSCORE", "float"),
    ("model_best_test_score", "MBTSCORE", "float"),
]

OFFLINE_TRAIN_LOG_FORMAT = [
    ("train_iteration", "I", "int"),
    ("epoch", "E", "int"),
    ("train_dataset_size", "TD", "int"),
    ("val_dataset_size", "VD", "int"),
    ("test_dataset_size", "testD", "int"),
    ("model_loss", "MLOSS", "float"),
    ("model_val_score", "MVSCORE", "float"),
    ("model_best_val_score", "MBVSCORE", "float"),
    ("model_test_score", "MTSCORE", "float"),
    ("model_best_test_score", "MBTSCORE", "float"),
]

class ModelTrainer:
    """Trainer for dynamics models.

    Args:
        model (:class:`mbrl.models.Model`): a model to train.
        optim_lr (float): the learning rate for the optimizer (using Adam).
        weight_decay (float): the weight decay to use.
        logger (:class:`mbrl.util.Logger`, optional): the logger to use.
    """

    _LOG_GROUP_NAME = "model_train"
    _LOG_GROUP_NAME_OFFLINE_TRAIN = "offline_model_train"

    def __init__(
        self,
        model: Model,
        optim_lr: float = 1e-4,
        weight_decay: float = 1e-5,
        optim_eps: float = 1e-8,
        logger: Optional[Logger] = None,
    ):
        self.model = model
        self._train_iteration = 0

        self.logger = logger
        if self.logger:
            self.logger.register_group(
                self._LOG_GROUP_NAME,
                MODEL_LOG_FORMAT,
                color="blue",
                dump_frequency=1,
            )
            self.logger.register_group(
                self._LOG_GROUP_NAME_OFFLINE_TRAIN,
                OFFLINE_TRAIN_LOG_FORMAT,
                color="red",
                dump_frequency=1,
            )

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=optim_lr,
            weight_decay=weight_decay,
            eps=optim_eps,
        )

    def train(
        self,
        train_using_only_offline_data,
        dataset_train: TransitionIterator,
        dataset_val: Optional[TransitionIterator] = None,
        dataset_test: Optional[TransitionIterator] = None,
        num_epochs: Optional[int] = None,
        patience: Optional[int] = None,
        improvement_threshold: float = 0.01,
        callback: Optional[Callable] = None,
        batch_callback: Optional[Callable] = None,
        evaluate: bool = True,
        test: bool = True,
        silent: bool = False,
    ) -> Tuple[List[float], List[float]]:
        """Trains the model for some number of epochs.

        This method iterates over the stored train dataset, one batch of transitions at a time,
        updates the model.

        If a validation dataset is provided in the constructor, this method will also evaluate
        the model over the validation data once per training epoch. The method will keep track
        of the weights with the best validation score, and after training the weights of the
        model will be set to the best weights. If no validation dataset is provided, the method
        will keep the model with the best loss over training data.

        Args:
            dataset_train (:class:`mbrl.util.TransitionIterator`): the iterator to
                use for the training data.
            dataset_val (:class:`mbrl.util.TransitionIterator`, optional):
                an iterator to use for the validation data.
            dataset_test (:class:`mbrl.util.TransitionIterator`, optional):
                an iterator to use for the test data.
            num_epochs (int, optional): if provided, the maximum number of epochs to train for.
                Default is ``None``, which indicates there is no limit.
            patience (int, optional): if provided, the patience to use for training. That is,
                training will stop after ``patience`` number of epochs without improvement.
                Ignored if ``evaluate=False`.
            improvement_threshold (float): The threshold in relative decrease of the evaluation
                score at which the model is seen as having improved.
                Ignored if ``evaluate=False`.
            callback (callable, optional): if provided, this function will be called after
                every training epoch with the following positional arguments::

                    - the model that's being trained
                    - total number of calls made to ``trainer.train()``
                    - current epoch
                    - training loss
                    - validation score (for ensembles, factored per member)
                    - best validation score so far

            batch_callback (callable, optional): if provided, this function will be called
                for every batch with the output of ``model.update()`` (during training),
                and ``model.eval_score()`` (during evaluation). It will be called
                with four arguments ``(epoch_index, loss/score, meta, mode)``, where
                ``mode`` is one of ``"train"`` or ``"eval"``, indicating if the callback
                was called during training or evaluation.

            evaluate (bool, optional): if ``True``, the trainer will use ``model.eval_score()``
                to keep track of the best model. If ``False`` the model will not compute
                an evaluation score, and simply train for some number of epochs. Defaults to
                ``True``.

            silent (bool): if ``True`` logging and progress bar are deactivated. Defaults
                to ``False``.

        Returns:
            (tuple of two list(float)): the history of training losses and validation losses.

        """
        eval_dataset = dataset_train if dataset_val is None else dataset_val
        test_dataset = dataset_train if dataset_test is None else dataset_test

        if self.model.model.model_type == "sgcn":
            if isinstance(eval_dataset, BootstrapIterator):
                eval_dataset.toggle_bootstrap()

            def visualize_and_log_adj_mat(adj_mat, title, step):
                adj_mat_np = adj_mat.cpu().numpy()
                fig, ax = plt.subplots()
                im = ax.imshow(adj_mat_np, cmap='binary')
                cbar = ax.figure.colorbar(im, ax=ax)
                wandb.log({title: wandb.Image(fig, caption="step:{}".format(step))})
                plt.close(fig)

            batch = next(eval_dataset)
            sgcn_adj_mat, real_adj_mat = self.model.get_two_adj_mat(batch)
            plot_interval = 10  # 设置绘图间隔，可以根据需要进行调整

            print("sgcn_adj_mat[0]", sgcn_adj_mat[0])
            print("real_adj_mat[0]", real_adj_mat[0])
            for i in range(sgcn_adj_mat.shape[0]):
                if i % plot_interval == 0:
                    visualize_and_log_adj_mat(sgcn_adj_mat[i], "sgcn_adj_mat", dataset_train.num_stored + i)
                    visualize_and_log_adj_mat(real_adj_mat[i], "real_adj_mat", dataset_train.num_stored + i)

            if isinstance(eval_dataset, BootstrapIterator):
                eval_dataset.toggle_bootstrap()

        training_losses, val_scores, test_scores = [], [], []
        best_weights: Optional[Dict] = None
        epoch_iter = range(num_epochs) if num_epochs else itertools.count()
        epochs_since_update = 0
        best_val_score = self.evaluate(eval_dataset) if evaluate else None
        best_test_score = self.evaluate(test_dataset) if test else None
        # only enable tqdm if training for a single epoch,
        # otherwise it produces too much output
        disable_tqdm = silent or (num_epochs is None or num_epochs > 1)

        for epoch in epoch_iter:
            if batch_callback:
                batch_callback_epoch = functools.partial(batch_callback, epoch)
            else:
                batch_callback_epoch = None
            batch_losses: List[float] = []
            for batch in tqdm.tqdm(dataset_train, disable=disable_tqdm):
                loss, meta = self.model.update(batch, self.optimizer)
                batch_losses.append(loss)
                if batch_callback_epoch:
                    batch_callback_epoch(loss, meta, "train")
            total_avg_loss = np.mean(batch_losses).mean().item()
            training_losses.append(total_avg_loss)

            eval_score = None
            model_val_score = 0
            if evaluate:
                eval_score = self.evaluate(
                    eval_dataset, batch_callback=batch_callback_epoch
                )
                val_scores.append(eval_score.mean().item())

                maybe_best_weights = self.maybe_get_best_weights(
                    best_val_score, eval_score, improvement_threshold
                )
                if maybe_best_weights:
                    best_val_score = torch.minimum(best_val_score, eval_score)
                    best_weights = maybe_best_weights
                    epochs_since_update = 0
                else:
                    epochs_since_update += 1
                model_val_score = eval_score.mean()

            test_score = None
            model_test_score = 0
            if test:
                test_score = self.evaluate(
                    test_dataset, batch_callback=batch_callback_epoch
                )
                test_scores.append(test_score.mean().item())

                maybe_best_weights = self.maybe_get_best_weights(
                    best_test_score, test_score, improvement_threshold
                )
                if maybe_best_weights:
                    best_test_score = torch.minimum(best_test_score, test_score)
                    best_weights = maybe_best_weights
                    epochs_since_update = 0
                else:
                    epochs_since_update += 1
                model_test_score = test_score.mean()

            if self.logger and not silent:
                log_data = {
                            "iteration": self._train_iteration,
                            "epoch": epoch,
                            "train_dataset_size": dataset_train.num_stored,
                            "val_dataset_size": dataset_val.num_stored
                            if dataset_val is not None
                            else 0,
                            "test_dataset_size": dataset_test.num_stored
                            if dataset_test is not None
                            else 0,
                            "model_loss": total_avg_loss,
                            "model_val_score": model_val_score,
                            "model_best_val_score": best_val_score.mean()
                            if best_val_score is not None
                            else 0,
                            "model_test_score": model_test_score,
                            "model_best_test_score": best_test_score.mean()
                            if best_test_score is not None
                            else 0,
                        }
                if train_using_only_offline_data:
                    self.logger.log_data(self._LOG_GROUP_NAME_OFFLINE_TRAIN, log_data)
                else:
                    self.logger.log_data(self._LOG_GROUP_NAME, log_data)
            if train_using_only_offline_data:
                wandb.log(
                    {
                        "offline_epoch": epoch,
                        "offline_train_dataset_size": dataset_train.num_stored,
                        "offline_val_dataset_size": dataset_val.num_stored
                        if dataset_val is not None
                        else 0,
                        "offline_test_dataset_size": dataset_test.num_stored
                        if dataset_test is not None
                        else 0,
                        "offline_model_loss": total_avg_loss,
                        "offline_model_val_score": model_val_score,
                        "offline_model_best_val_score": best_val_score.mean()
                        if best_val_score is not None
                        else 0,
                        "offline_model_test_score": model_test_score,
                        "offline_model_best_test_score": best_test_score.mean()
                        if best_test_score is not None
                        else 0,
                    }
                )
            else:
                wandb.log(
                    {
                        "iteration": self._train_iteration,
                        "epoch": epoch,
                        "train_dataset_size": dataset_train.num_stored,
                        "val_dataset_size": dataset_val.num_stored
                        if dataset_val is not None
                        else 0,
                        "test_dataset_size": dataset_test.num_stored
                        if dataset_test is not None
                        else 0,
                        "model_loss": total_avg_loss,
                        "model_val_score": model_val_score,
                        "model_best_val_score": best_val_score.mean()
                        if best_val_score is not None
                        else 0,
                        "model_test_score": model_test_score,
                        "model_best_test_score": best_test_score.mean()
                        if best_test_score is not None
                        else 0,
                    }
                )

            MODEL_LOG_FORMAT = [
                ("train_iteration", "I", "int"),
                ("epoch", "E", "int"),
                ("train_dataset_size", "TD", "int"),
                ("val_dataset_size", "VD", "int"),
                ("test_dataset_size", "testD", "int"),
                ("model_loss", "MLOSS", "float"),
                ("model_val_score", "MVSCORE", "float"),
                ("model_best_val_score", "MBVSCORE", "float"),
                ("model_test_score", "MTSCORE", "float"),
                ("model_best_test_score", "MBTSCORE", "float"),
            ]

            if callback:
                callback(
                    self.model,
                    self._train_iteration,
                    epoch,
                    total_avg_loss,
                    eval_score,
                    best_val_score,
                )

            if patience and epochs_since_update >= patience:
                break

        # saving the best models:
        if evaluate:
            self._maybe_set_best_weights_and_elite(best_weights, best_val_score)

        self._train_iteration += 1
        return training_losses, val_scores

    def evaluate(
        self, dataset: TransitionIterator, batch_callback: Optional[Callable] = None
    ) -> torch.Tensor:
        """Evaluates the model on the validation dataset.

        Iterates over the dataset, one batch at a time, and calls
        :meth:`mbrl.models.Model.eval_score` to compute the model score
        over the batch. The method returns the average score over the whole dataset.

        Args:
            dataset (bool): the transition iterator to use.
            batch_callback (callable, optional): if provided, this function will be called
                for every batch with the output of ``model.eval_score()`` (the score will
                be passed as a float, reduced using mean()). It will be called
                with four arguments ``(epoch_index, loss/score, meta, mode)``, where
                ``mode`` is the string ``"eval"``.

        Returns:
            (tensor): The average score of the model over the dataset (and for ensembles, per
                ensemble member).
        """
        if isinstance(dataset, BootstrapIterator):
            dataset.toggle_bootstrap()

        batch_scores_list = []
        for batch in dataset:
            batch_score, meta = self.model.eval_score(batch)
            batch_scores_list.append(batch_score)
            if batch_callback:
                batch_callback(batch_score.mean(), meta, "eval")
        try:
            batch_scores = torch.cat(
                batch_scores_list, dim=batch_scores_list[0].ndim - 2
            )
        except RuntimeError as e:
            print(
                f"There was an error calling ModelTrainer.evaluate(). "
                f"Note that model.eval_score() should be non-reduced. Error was: {e}"
            )
            raise e
        if isinstance(dataset, BootstrapIterator):
            dataset.toggle_bootstrap()

        mean_axis = 1 if batch_scores.ndim == 2 else (1, 2)
        batch_scores = batch_scores.mean(dim=mean_axis)

        return batch_scores

    def maybe_get_best_weights(
        self,
        best_val_score: torch.Tensor,
        val_score: torch.Tensor,
        threshold: float = 0.01,
    ) -> Optional[Dict]:
        """Return the current model state dict  if the validation score improves.

        For ensembles, this checks the validation for each ensemble member separately.

        Args:
            best_val_score (tensor): the current best validation losses per model.
            val_score (tensor): the new validation loss per model.
            threshold (float): the threshold for relative improvement.

        Returns:
            (dict, optional): if the validation score's relative improvement over the
            best validation score is higher than the threshold, returns the state dictionary
            of the stored model, otherwise returns ``None``.
        """
        improvement = (best_val_score - val_score) / torch.abs(best_val_score)
        improved = (improvement > threshold).any().item()
        return copy.deepcopy(self.model.state_dict()) if improved else None

    def _maybe_set_best_weights_and_elite(
        self, best_weights: Optional[Dict], best_val_score: torch.Tensor
    ):
        if best_weights is not None:
            self.model.load_state_dict(best_weights)
        if len(best_val_score) > 1 and hasattr(self.model, "num_elites"):
            sorted_indices = np.argsort(best_val_score.tolist())
            elite_models = sorted_indices[: self.model.num_elites]
            self.model.set_elite(elite_models)
