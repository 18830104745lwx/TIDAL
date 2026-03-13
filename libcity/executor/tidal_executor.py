"""
TIDALExecutor — 专用于 TIDAL 模型的精简 Executor

相比 TrafficFormerExecutor:
  - 移除了不必要的 Laplacian 位置编码计算
  - 移除了未使用的 DTW / 深监督逻辑
  - 仅在基类基础上添加 CosineLR warmup 调度器支持
"""
import time
import numpy as np
import torch
import os

from ray import tune
from libcity.executor.scheduler import CosineLRScheduler
from libcity.executor.traffic_state_executor import TrafficStateExecutor
from libcity.utils import reduce_array


class TIDALExecutor(TrafficStateExecutor):

    def __init__(self, config, model):
        self.lr_warmup_epoch = config.get("lr_warmup_epoch", 5)
        self.lr_warmup_init = config.get("lr_warmup_init", 1e-6)
        super().__init__(config, model)

    # ------------------------------------------------------------------
    # 重写: 添加 AdamW + CosineLR 支持
    # ------------------------------------------------------------------
    def _build_optimizer(self):
        self._logger.info('You select `{}` optimizer.'.format(self.learner.lower()))
        if self.learner.lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.learning_rate,
                eps=self.lr_epsilon, betas=self.lr_betas,
                weight_decay=self.weight_decay)
        else:
            optimizer = super()._build_optimizer()
        return optimizer

    def _build_lr_scheduler(self):
        if self.lr_decay and self.lr_scheduler_type.lower() == 'cosinelr':
            self._logger.info('You select `cosinelr` lr_scheduler.')
            lr_scheduler = CosineLRScheduler(
                self.optimizer, t_initial=self.epochs,
                lr_min=self.lr_eta_min, decay_rate=self.lr_decay_ratio,
                warmup_t=self.lr_warmup_epoch, warmup_lr_init=self.lr_warmup_init)
            return lr_scheduler
        return super()._build_lr_scheduler()

    # ------------------------------------------------------------------
    # 重写: train 循环以正确处理 CosineLR 的 step 逻辑
    # ------------------------------------------------------------------
    def train(self, train_dataloader, eval_dataloader):
        self._logger.info('Start training ...')
        min_val_loss = float('inf')
        wait = 0
        best_epoch = 0
        train_time = []
        eval_time = []
        num_batches = len(train_dataloader)
        self._logger.info("num_batches:{}".format(num_batches))

        batches_seen = num_batches * self._epoch_num
        for epoch_idx in range(self._epoch_num, self.epochs):
            start_time = time.time()
            losses, batches_seen = self._train_epoch(
                train_dataloader, epoch_idx, batches_seen, self.loss_func)
            t1 = time.time()
            train_time.append(t1 - start_time)
            train_loss = np.mean(losses)
            if self.distributed:
                train_loss = reduce_array(train_loss, self.world_size, self.device)
            self._writer.add_scalar('training loss', train_loss, batches_seen)
            self._logger.info("epoch complete!")

            self._logger.info("evaluating now!")
            t2 = time.time()
            val_loss = self._valid_epoch(
                eval_dataloader, epoch_idx, batches_seen, self.loss_func)
            end_time = time.time()
            eval_time.append(end_time - t2)

            epoch_time = end_time - start_time
            if self.distributed:
                epoch_time = reduce_array(np.array(epoch_time), self.world_size, self.device)

            # lr_scheduler step
            if self.lr_scheduler is not None:
                if self.lr_scheduler_type.lower() == 'reducelronplateau':
                    self.lr_scheduler.step(val_loss)
                elif self.lr_scheduler_type.lower() == 'cosinelr':
                    self.lr_scheduler.step(epoch_idx + 1)
                else:
                    self.lr_scheduler.step()

            if (epoch_idx % self.log_every) == 0:
                log_lr = self.optimizer.param_groups[0]['lr']
                message = ('Epoch [{}/{}] ({}) train_loss: {:.4f}, val_loss: {:.4f}, '
                           'lr: {:.6f}, {:.2f}s').format(
                    epoch_idx, self.epochs, batches_seen,
                    train_loss, val_loss, log_lr, epoch_time)
                self._logger.info(message)

            if self.hyper_tune:
                with tune.checkpoint_dir(step=epoch_idx) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    self.save_model(path)
                tune.report(loss=val_loss)

            if val_loss < min_val_loss:
                wait = 0
                if self.saved:
                    model_file_name = self.save_model_with_epoch(epoch_idx)
                    self._logger.info(
                        'Val loss decrease from {:.4f} to {:.4f}, '
                        'saving to {}'.format(min_val_loss, val_loss, model_file_name))
                min_val_loss = val_loss
                best_epoch = epoch_idx
            else:
                wait += 1
                if wait == self.patience and self.use_early_stop:
                    self._logger.warning('Early stopping at epoch: %d' % epoch_idx)
                    break

        if len(train_time) > 0:
            average_train_time = sum(train_time) / len(train_time)
            average_eval_time = sum(eval_time) / len(eval_time)
            if self.distributed:
                average_train_time = reduce_array(average_train_time, self.world_size, self.device)
                average_eval_time = reduce_array(average_eval_time, self.world_size, self.device)
            self._logger.info(
                'Trained totally {} epochs, average train time is {:.3f}s, '
                'average eval time is {:.3f}s'.format(
                    len(train_time), average_train_time, average_eval_time))
        if self.load_best_epoch:
            self.load_model_with_epoch(best_epoch)
        return min_val_loss

    def _train_epoch(self, train_dataloader, epoch_idx, batches_seen=None, loss_func=None):
        self.model.train()
        loss_func = loss_func if loss_func is not None else self.model.calculate_loss
        losses = []
        for batch in train_dataloader:
            batch.to_tensor(self.device)
            loss = loss_func(batch)
            self._logger.debug(loss.item())
            losses.append(loss.item())
            batches_seen += 1
            loss = loss / self.grad_accmu_steps
            loss.backward()
            if self.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            if batches_seen % self.grad_accmu_steps == 0:
                self.optimizer.step()
                if self.lr_scheduler is not None:
                    if self.lr_scheduler_type.lower() == 'cosinelr':
                        self.lr_scheduler.step_update(num_updates=batches_seen)
                self.optimizer.zero_grad()
        return losses, batches_seen
