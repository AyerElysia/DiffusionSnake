import time
import datetime
import torch
import tqdm


class Trainer(object):
    def __init__(self, network):   # 这里的 network 指 lib/train/trainers/snake.py 中的 NetworkWrapper 对象（本质上是一个封装好了损失函数的网络模型）
        current_device = torch.cuda.current_device()
        print(f"当前默认的 GPU 设备索引: {current_device}")
        network = network.cuda()   # 单卡训练直接将模型迁移到 GPU

        self.network = network

    def reduce_loss_stats(self, loss_stats):
        reduced_losses = {k: torch.mean(v.float()) for k, v in loss_stats.items()}
        return reduced_losses

    def to_cuda(self, batch):
        for k, v in list(batch.items()):
            if k in ('meta', 'orig_img', 'img_path') or k == 'locate_feat' or str(k).startswith('locate_feat_'):
                continue
            # list/tuple of tensors
            if isinstance(v, (list, tuple)):
                moved = []
                for itm in v:
                    if torch.is_tensor(itm):
                        moved.append(itm.cuda())
                    else:
                        moved.append(itm)
                batch[k] = moved if isinstance(v, list) else tuple(moved)
            elif torch.is_tensor(v):
                batch[k] = v.cuda()
            else:
                # leave non-tensors as-is (e.g., scalars, strings)
                batch[k] = v
        return batch

    def train(self, epoch, data_loader, optimizer, recorder, json_logger=None):
        max_iter = len(data_loader)
        print(len(data_loader))
        self.network.train()  # 将模型设置为训练模式
        end = time.time()
        for iteration, batch in enumerate(data_loader):
            data_time = time.time() - end
            iteration = iteration + 1
            recorder.step += 1

            batch = self.to_cuda(batch)
            # 附加epoch/iteration元信息，并在每个epoch的第一个iteration触发可视化
            try:
                if 'meta' not in batch or batch['meta'] is None:
                    batch['meta'] = {}
                batch['meta'].update({
                    'epoch': epoch,
                    'iteration': iteration,
                    'max_iter': len(data_loader),
                    'save_vis': (iteration == 1)
                })
            except Exception:
                pass

            output, loss, loss_stats, image_stats = self.network(batch)   # 执行 NetworkWrapper 对象的 forward 函数
            # 上句，这里有没有可能定义两个loss : loss1 & loss2 , 分别记录darnet和dla两个输出端的损失
            # 在误差逆传播的过程中分别训练两个头的参数（更新一个头的时候mask掉另一个）————训练方案有待讨论

            # training stage: loss; optimizer; scheduler
            loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.network.parameters(), 40)
            optimizer.step()

            # data recording stage: loss_stats, time, image_stats
            loss_stats = self.reduce_loss_stats(loss_stats)
            recorder.update_loss_stats(loss_stats)

            batch_time = time.time() - end
            end = time.time()
            recorder.batch_time.update(batch_time)
            recorder.data_time.update(data_time)

            if iteration % 20 == 0 or iteration == (max_iter - 1):
                # print training state
                eta_seconds = recorder.batch_time.global_avg * (max_iter - iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                lr = optimizer.param_groups[0]['lr']
                memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0

                training_state = '  '.join(['eta: {}', '{}', 'lr: {:.6f}', 'max_mem: {:.0f}'])
                training_state = training_state.format(eta_string, str(recorder), lr, memory)
                print(training_state)

                if json_logger is not None:
                    try:
                        loss_dict = {}
                        for k, v in recorder.loss_stats.items():
                            if hasattr(v, 'avg'):
                                loss_dict[k] = float(v.avg)
                            else:
                                try:
                                    loss_dict[k] = float(v)
                                except Exception:
                                    pass
                        json_logger.log({
                            'epoch': int(epoch),
                            'iteration': int(iteration),
                            'max_iter': int(max_iter),
                            'lr': float(lr),
                            'max_mem_mb': float(memory),
                            'data_time': float(recorder.data_time.avg),
                            'batch_time': float(recorder.batch_time.avg),
                            'loss': loss_dict,
                            # GPU memory tracking
                            'cuda_alloc_mb': float(torch.cuda.memory_allocated() / 1024.0 / 1024.0),
                            'cuda_reserved_mb': float(torch.cuda.memory_reserved() / 1024.0 / 1024.0)
                        })
                    except Exception:
                        pass

                # record loss_stats and image_dict
                recorder.update_image_stats(image_stats)
                recorder.record('train')

            # 释放当前迭代的临时张量引用（不再调用 empty_cache，避免每步 50-200ms 开销）
            del output, loss, loss_stats, image_stats, batch

    def train_staged_with_tracking(self, epoch, data_loader, optimizer_groups, recorder, loss_tracker):
        """
        分阶段训练：接收多个优化器组，统一前向/反向，然后分别 step。
        在每个 iteration 统计损失并写入 recorder；在每个 epoch 结束由上层调用 loss_tracker.end_epoch。
        optimizer_groups: List[{'name': str, 'optimizer': torch.optim.Optimizer}]
        """
        max_iter = len(data_loader)
        self.network.train()
        end = time.time()

        for iteration, batch in enumerate(data_loader):
            data_time = time.time() - end
            iteration = iteration + 1
            recorder.step += 1

            batch = self.to_cuda(batch)
            # zero all optimizers
            for og in optimizer_groups:
                og['optimizer'].zero_grad()

            # 附加epoch/iteration元信息，并在每个epoch的第一个iteration触发可视化
            try:
                if 'meta' not in batch or batch['meta'] is None:
                    batch['meta'] = {}
                batch['meta'].update({
                    'epoch': epoch,
                    'iteration': iteration,
                    'max_iter': len(data_loader),
                    'save_vis': (iteration == 1)
                })
            except Exception:
                pass

            output, loss, loss_stats, image_stats = self.network(batch)
            loss = loss.mean()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.network.parameters(), 40)

            # step all optimizers
            for og in optimizer_groups:
                og['optimizer'].step()

            # logging
            loss_stats = self.reduce_loss_stats(loss_stats)
            recorder.update_loss_stats(loss_stats)
            # send to loss tracker (average per-iteration; epoch-averaging will be done in end_epoch)
            try:
                # convert tensors to floats where needed
                loss_dict = {k: float(v.detach().cpu().item()) if hasattr(v, 'detach') else float(v) for k, v in loss_stats.items()}
                loss_tracker.update(loss_dict)
            except Exception:
                pass

            batch_time = time.time() - end
            end = time.time()
            recorder.batch_time.update(batch_time)
            recorder.data_time.update(data_time)

            if iteration % 20 == 0 or iteration == (max_iter - 1):
                eta_seconds = recorder.batch_time.global_avg * (max_iter - iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                # show the first group's lr for readability
                lr = optimizer_groups[0]['optimizer'].param_groups[0]['lr'] if optimizer_groups else 0.0
                memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0

                training_state = '  '.join(['eta: {}', '{}', 'lr: {:.6f}', 'max_mem: {:.0f}'])
                training_state = training_state.format(eta_string, str(recorder), lr, memory)
                print(training_state)

                recorder.update_image_stats(image_stats)
                recorder.record('train')

    def val(self, epoch, data_loader, evaluator=None, recorder=None):
        self.network.eval()
        torch.cuda.empty_cache()
        val_loss_stats = {}
        data_size = len(data_loader)
        for batch in tqdm.tqdm(data_loader):
            # move only tensors to CUDA, keep others on CPU
            for k, v in list(batch.items()):
                if k in ('meta', 'orig_img', 'img_path') or k == 'locate_feat' or str(k).startswith('locate_feat_'):
                    continue
                if isinstance(v, (list, tuple)):
                    moved = []
                    for itm in v:
                        if torch.is_tensor(itm):
                            moved.append(itm.cuda())
                        else:
                            moved.append(itm)
                    batch[k] = moved if isinstance(v, list) else tuple(moved)
                elif torch.is_tensor(v):
                    batch[k] = v.cuda()
                else:
                    batch[k] = v

            with torch.no_grad():
                output, loss, loss_stats, image_stats = self.network(batch)
                if evaluator is not None:
                    evaluator.evaluate(output, batch)

            loss_stats = self.reduce_loss_stats(loss_stats)
            for k, v in loss_stats.items():
                val_loss_stats.setdefault(k, 0)
                val_loss_stats[k] += v

        loss_state = []
        for k in val_loss_stats.keys():
            val_loss_stats[k] /= data_size
            loss_state.append('{}: {:.4f}'.format(k, val_loss_stats[k]))
        print(loss_state)



        if recorder:
            recorder.record('val', epoch, val_loss_stats, image_stats)
