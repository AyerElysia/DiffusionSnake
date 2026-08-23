import datetime
import os
import time
from contextlib import nullcontext

import torch
import tqdm


class Trainer(object):
    def __init__(self, network):
        current_device = torch.cuda.current_device()
        print(f"当前默认的 GPU 设备索引: {current_device}")
        self.network = network.cuda()

        self.amp_enabled = False
        self.amp_dtype = torch.float16
        self.grad_scaler = None
        self.gradient_clip = 40.0
        self.gradient_accumulation_steps = 1
        self.empty_cache_interval = 0
        self.rank = 0
        self.is_main_process = True
        self.log_interval = 20
        self.memory_debug_steps = max(
            int(os.environ.get('CUDA_MEMORY_DEBUG_STEPS', '0') or 0), 0
        )
        self.memory_debug_interval = max(
            int(os.environ.get('CUDA_MEMORY_DEBUG_INTERVAL', '1') or 1), 1
        )
        self.memory_debug_all_ranks = (
            os.environ.get('CUDA_MEMORY_DEBUG_ALL_RANKS', '').strip().lower()
            in ('1', 'true', 'yes')
        )

    def configure_runtime(
            self,
            amp_enabled=False,
            amp_dtype=torch.float16,
            grad_scaler=None,
            gradient_clip=40.0,
            gradient_accumulation_steps=1,
            empty_cache_interval=0,
            rank=0,
            is_main_process=True):
        """Configure runtime-only knobs after model construction."""
        gradient_clip = 0.0 if gradient_clip is None else float(gradient_clip)
        if gradient_clip < 0.0:
            raise ValueError(f"gradient_clip must be >= 0, got {gradient_clip}")
        gradient_accumulation_steps = int(gradient_accumulation_steps)
        if gradient_accumulation_steps < 1:
            raise ValueError(
                f"gradient_accumulation_steps must be >= 1, got {gradient_accumulation_steps}"
            )
        empty_cache_interval = int(empty_cache_interval)
        if empty_cache_interval < 0:
            raise ValueError(
                f"empty_cache_interval must be >= 0, got {empty_cache_interval}"
            )

        self.amp_enabled = bool(amp_enabled)
        self.amp_dtype = amp_dtype
        self.grad_scaler = grad_scaler
        self.gradient_clip = gradient_clip
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.empty_cache_interval = empty_cache_interval
        self.rank = int(rank)
        self.is_main_process = bool(is_main_process)

    def _runtime_device(self):
        try:
            return next(self.network.parameters()).device
        except StopIteration:
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _memory_debug_enabled(self, iteration):
        return (
            torch.cuda.is_available()
            and self.memory_debug_steps > 0
            and int(iteration) <= self.memory_debug_steps
            and int(iteration) % self.memory_debug_interval == 0
            and (self.is_main_process or self.memory_debug_all_ranks)
        )

    @staticmethod
    def _tensor_megabytes(value):
        return float(value.numel() * value.element_size()) / (1024.0 ** 2)

    def _batch_memory_details(self, batch):
        if not isinstance(batch, dict):
            return 'contours=? locate_mb=?'

        contours = '?'
        ct_01 = batch.get('ct_01')
        if torch.is_tensor(ct_01):
            contours = int(ct_01.detach().sum().item())

        locate_mb = 0.0
        locate_feat = batch.get('locate_feat')
        if torch.is_tensor(locate_feat):
            locate_mb = self._tensor_megabytes(locate_feat)
        elif isinstance(locate_feat, (list, tuple)):
            locate_mb = sum(
                self._tensor_megabytes(value)
                for value in locate_feat
                if torch.is_tensor(value)
            )
        return 'contours={} locate_mb={:.1f}'.format(contours, locate_mb)

    def _log_cuda_memory(
            self,
            epoch,
            iteration,
            phase,
            batch=None):
        if not self._memory_debug_enabled(iteration):
            return
        device = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(device) / (1024.0 ** 2)
        reserved = torch.cuda.memory_reserved(device) / (1024.0 ** 2)
        peak = torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
        details = self._batch_memory_details(batch) if batch is not None else ''
        print(
            '[cuda-mem] rank={} epoch={} iter={} phase={} '
            'alloc_mb={:.1f} reserved_mb={:.1f} peak_mb={:.1f} {}'.format(
                self.rank,
                epoch,
                iteration,
                phase,
                allocated,
                reserved,
                peak,
                details,
            ),
            flush=True,
        )

    def reduce_loss_stats(self, loss_stats):
        device = self._runtime_device()
        reduced_losses = {}
        for key, value in (loss_stats or {}).items():
            if torch.is_tensor(value):
                reduced_losses[key] = value.float().mean()
            else:
                reduced_losses[key] = torch.tensor(float(value), device=device)
        return reduced_losses

    @staticmethod
    def _loss_stats_to_host(loss_stats):
        """Move all scalar statistics with one synchronization point."""
        if not loss_stats:
            return {}

        keys = list(loss_stats.keys())
        values = []
        target_device = None
        for key in keys:
            value = loss_stats[key]
            if torch.is_tensor(value):
                value = value.detach().float()
                if value.numel() != 1:
                    value = value.mean()
                if value.is_cuda:
                    target_device = value.device
                values.append(value)
            else:
                values.append(float(value))

        if target_device is None:
            return {key: float(value) for key, value in zip(keys, values)}

        packed = []
        for value in values:
            if torch.is_tensor(value):
                packed.append(value.to(device=target_device))
            else:
                packed.append(torch.tensor(float(value), device=target_device))
        host_values = torch.stack(packed).cpu().tolist()
        return {key: float(value) for key, value in zip(keys, host_values)}

    def to_cuda(self, batch):
        for key, value in list(batch.items()):
            if key in ('meta', 'orig_img', 'img_path'):
                continue

            if key == 'locate_feat' and isinstance(value, (list, tuple)):
                moved = [
                    item.cuda(non_blocking=True)
                    if torch.is_tensor(item)
                    else torch.as_tensor(item).cuda(non_blocking=True)
                    for item in value
                ]
                batch[key] = moved if isinstance(value, list) else tuple(moved)
                continue

            if isinstance(value, (list, tuple)):
                moved = [
                    item.cuda(non_blocking=True) if torch.is_tensor(item) else item
                    for item in value
                ]
                batch[key] = moved if isinstance(value, list) else tuple(moved)
            elif torch.is_tensor(value):
                batch[key] = value.cuda(non_blocking=True)
        return batch

    def train(self, epoch, data_loader, optimizer, recorder, json_logger=None):
        max_iter = len(data_loader)
        if self.is_main_process:
            print(f'train steps: {max_iter}', flush=True)
        self.network.train()
        end = time.time()
        accumulation_steps = self.gradient_accumulation_steps

        for iteration, batch in enumerate(data_loader, start=1):
            data_time = time.time() - end
            recorder.step += 1
            batch = self.to_cuda(batch)
            self._log_cuda_memory(epoch, iteration, 'after_cuda', batch=batch)

            window_index = (iteration - 1) % accumulation_steps
            window_start = iteration - window_index
            window_end = min(window_start + accumulation_steps - 1, max_iter)
            window_size = window_end - window_start + 1
            is_optimizer_step = iteration == window_end
            if window_index == 0:
                optimizer.zero_grad(set_to_none=True)

            if isinstance(batch, dict):
                meta = batch.get('meta')
                if not isinstance(meta, dict):
                    meta = {}
                    batch['meta'] = meta
                meta.update({
                    'epoch': epoch,
                    'iteration': iteration,
                    'max_iter': max_iter,
                    'save_vis': iteration == 1,
                })

            # DDP synchronizes gradients through autograd hooks. Do not add a
            # second full-model all_reduce after backward.
            sync_context = nullcontext()
            if (
                    accumulation_steps > 1
                    and not is_optimizer_step
                    and hasattr(self.network, 'no_sync')):
                sync_context = self.network.no_sync()

            with sync_context:
                with torch.cuda.amp.autocast(
                        enabled=self.amp_enabled,
                        dtype=self.amp_dtype):
                    output, loss, loss_stats, image_stats = self.network(batch)
                    loss = loss.mean()
                    backward_loss = loss / float(window_size)
                self._log_cuda_memory(epoch, iteration, 'after_forward', batch=batch)

                if self.grad_scaler is not None:
                    self.grad_scaler.scale(backward_loss).backward()
                else:
                    backward_loss.backward()
                self._log_cuda_memory(epoch, iteration, 'after_backward', batch=batch)

            if is_optimizer_step:
                if self.grad_scaler is not None:
                    self.grad_scaler.unscale_(optimizer)
                if self.gradient_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        self.network.parameters(), self.gradient_clip
                    )
                if self.grad_scaler is not None:
                    self.grad_scaler.step(optimizer)
                    self.grad_scaler.update()
                else:
                    optimizer.step()
                self._log_cuda_memory(epoch, iteration, 'after_step', batch=batch)

            batch_time = time.time() - end
            end = time.time()
            should_log = (
                self.is_main_process
                and (
                    iteration % self.log_interval == 0
                    or iteration == max_iter
                )
            )
            if should_log:
                reduced_stats = self.reduce_loss_stats(loss_stats)
                host_stats = self._loss_stats_to_host(reduced_stats)
                recorder.update_loss_stats(host_stats)
                recorder.batch_time.update(float(batch_time))
                recorder.data_time.update(float(data_time))

                eta_seconds = recorder.batch_time.global_avg * (max_iter - iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                lr = optimizer.param_groups[0]['lr']
                memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
                training_state = '  '.join(
                    ['eta: {}', '{}', 'lr: {:.6f}', 'max_mem: {:.0f}']
                ).format(eta_string, str(recorder), lr, memory)
                print(training_state, flush=True)

                if json_logger is not None:
                    loss_dict = {
                        key: float(value.avg)
                        for key, value in recorder.loss_stats.items()
                    }
                    json_logger.log({
                        'epoch': int(epoch),
                        'iteration': int(iteration),
                        'max_iter': int(max_iter),
                        'lr': float(lr),
                        'max_mem_mb': float(memory),
                        'data_time': float(recorder.data_time.avg),
                        'batch_time': float(recorder.batch_time.avg),
                        'loss': loss_dict,
                        'cuda_alloc_mb': float(
                            torch.cuda.memory_allocated() / 1024.0 / 1024.0
                        ),
                        'cuda_reserved_mb': float(
                            torch.cuda.memory_reserved() / 1024.0 / 1024.0
                        ),
                    })

                recorder.update_image_stats(image_stats)
                recorder.record('train')

            del output, loss, backward_loss, loss_stats, image_stats, batch
            self._log_cuda_memory(epoch, iteration, 'after_cleanup')
            if (
                    self.empty_cache_interval > 0
                    and iteration % self.empty_cache_interval == 0):
                torch.cuda.empty_cache()
                self._log_cuda_memory(epoch, iteration, 'after_empty_cache')

    def train_staged_with_tracking(
            self, epoch, data_loader, optimizer_groups, recorder, loss_tracker):
        """Train a staged setup used by legacy single-process experiments."""
        max_iter = len(data_loader)
        self.network.train()
        end = time.time()

        for iteration, batch in enumerate(data_loader, start=1):
            data_time = time.time() - end
            recorder.step += 1
            batch = self.to_cuda(batch)
            for group in optimizer_groups:
                group['optimizer'].zero_grad(set_to_none=True)

            if isinstance(batch, dict):
                meta = batch.get('meta')
                if not isinstance(meta, dict):
                    meta = {}
                    batch['meta'] = meta
                meta.update({
                    'epoch': epoch,
                    'iteration': iteration,
                    'max_iter': max_iter,
                    'save_vis': iteration == 1,
                })

            output, loss, loss_stats, image_stats = self.network(batch)
            loss = loss.mean()
            loss.backward()
            if self.gradient_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.gradient_clip
                )
            for group in optimizer_groups:
                group['optimizer'].step()

            loss_stats = self.reduce_loss_stats(loss_stats)
            recorder.update_loss_stats(self._loss_stats_to_host(loss_stats))
            try:
                loss_tracker.update({
                    key: float(value)
                    for key, value in self._loss_stats_to_host(loss_stats).items()
                })
            except Exception:
                pass

            batch_time = time.time() - end
            end = time.time()
            recorder.batch_time.update(float(batch_time))
            recorder.data_time.update(float(data_time))

            if iteration % self.log_interval == 0 or iteration == max_iter:
                eta_seconds = recorder.batch_time.global_avg * (max_iter - iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                lr = (
                    optimizer_groups[0]['optimizer'].param_groups[0]['lr']
                    if optimizer_groups else 0.0
                )
                memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
                print(
                    '  '.join(
                        ['eta: {}', '{}', 'lr: {:.6f}', 'max_mem: {:.0f}']
                    ).format(eta_string, str(recorder), lr, memory),
                    flush=True,
                )
                recorder.update_image_stats(image_stats)
                recorder.record('train')

            del output, loss, loss_stats, image_stats, batch

    def val(self, epoch, data_loader, evaluator=None, recorder=None):
        self.network.eval()
        torch.cuda.empty_cache()
        val_loss_stats = {}
        image_stats = {}
        data_size = len(data_loader)

        for batch in tqdm.tqdm(data_loader):
            batch = self.to_cuda(batch)
            with torch.no_grad():
                output, loss, loss_stats, image_stats = self.network(batch)
                if evaluator is not None:
                    evaluator.evaluate(output, batch)

            reduced = self.reduce_loss_stats(loss_stats)
            for key, value in reduced.items():
                val_loss_stats.setdefault(key, 0.0)
                val_loss_stats[key] += float(value.detach().cpu())

        if data_size:
            for key in val_loss_stats:
                val_loss_stats[key] /= data_size
        print([
            '{}: {:.4f}'.format(key, value)
            for key, value in val_loss_stats.items()
        ])

        if recorder:
            recorder.record('val', epoch, val_loss_stats, image_stats)
