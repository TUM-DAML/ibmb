import logging
import os
import time
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import f1_score
from tqdm import tqdm

from .prefetch_generators import get_prefetch_generator
from .train_utils import run_batch, MyGraph

# from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(self,
                 mode: str,
                 neighbor_sampling: str,
                 num_batches: list,
                 micro_batch: int = 1,
                 batch_size: int = 1,
                 epoch_max: int = 800,
                 epoch_min: int = 300,
                 patience: int = 100,
                 device: str = 'cuda',
                 notebook: bool = True):

        super().__init__()

        self.mode = mode
        self.neighbor_sampling = neighbor_sampling
        self.device = device
        self.notebook = notebook
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.micro_batch = micro_batch
        self.epoch_max = epoch_max
        self.epoch_min = epoch_min
        self.patience = patience

        self.database = defaultdict(list)

    def get_loss_scaling(self, len_loader: int):
        micro_batch = int(min(self.micro_batch, len_loader))
        num_batches = len_loader // self.batch_size + ((len_loader % self.batch_size) > 0)
        loss_scaling_lst = [micro_batch] * (num_batches // micro_batch) + [num_batches % micro_batch]
        return loss_scaling_lst, micro_batch

    def train(self,
              dataset,
              model,
              lr,
              reg,
              train_nodes=None,
              val_nodes=None,
              comment='',
              run_no=''):

        #         writer = SummaryWriter('./runs')
        patience_count = 0
        best_accs = {'train': 0., 'self': 0., 'part': 0., 'ppr': 0.}
        best_val_acc = 0.

        pbar = tqdm(range(self.epoch_max)) if self.notebook else range(self.epoch_max)
        np.random.seed(2021)

        model_dir = os.path.join('/nfs/students/qian', comment)
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)
        model_path = os.path.join(model_dir, f'model_{run_no}.pt')

        # start training
        training_curve = defaultdict(list)

        opt = torch.optim.Adam(model.p_list, lr=lr, weight_decay=reg)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.33, patience=30,
                                                               cooldown=10, min_lr=1e-4)

        dataset.set_split('train')
        next_loader = get_prefetch_generator(self.mode,
                                             self.neighbor_sampling,
                                             dataset,
                                             train_nodes,
                                             self.batch_size)
        for epoch in pbar:
            data_dic = {'self': {'loss': 0., 'acc': 0., 'num': 0},
                        'part': {'loss': 0., 'acc': 0., 'num': 0},
                        'train': {'loss': 0., 'acc': 0., 'num': 0},
                        'ppr': {'loss': 0., 'acc': 0., 'num': 0}, }

            update_count = 0

            # train
            model.train()
            loss_scaling_lst, cur_micro_batch = self.get_loss_scaling(len(dataset.train_loader))
            loader, next_loader = next_loader, None

            start_time = time.time()
            while True:
                data = loader.next()
                if data is None:
                    loader = None
                    loss = None
                    corrects = None
                    break
                else:
                    if data[1]:  # stop signal
                        successful_part = dataset.set_split('val_part')
                        if successful_part:
                            next_loader = get_prefetch_generator(self.mode,
                                                                 self.neighbor_sampling,
                                                                 dataset,
                                                                 val_nodes,
                                                                 batch_size=self.batch_size)
                        else:
                            successful_ppr = dataset.set_split('val_ppr')
                            if successful_ppr:
                                next_loader = get_prefetch_generator(self.mode,
                                                                     self.neighbor_sampling,
                                                                     dataset,
                                                                     val_nodes,
                                                                     batch_size=self.batch_size)
                            else:
                                dataset.set_split('val_self')
                                next_loader = get_prefetch_generator(self.mode,
                                                                     self.neighbor_sampling,
                                                                     dataset,
                                                                     val_nodes,
                                                                     batch_size=self.batch_size)

                loss, corrects, num_nodes, _, _ = run_batch(model, MyGraph(*data[0]), loss_scaling_lst[0],
                                                            verbose=False)
                data_dic['train']['loss'] += loss
                data_dic['train']['acc'] += corrects
                data_dic['train']['num'] += num_nodes
                update_count += 1

                if update_count >= cur_micro_batch:
                    loss_scaling_lst.pop(0)
                    opt.step()
                    opt.zero_grad()
                    update_count = 0

            # remainder
            if update_count:
                opt.step()
                opt.zero_grad()

            train_time = time.time() - start_time

            logging.info(f'\n allocated: {torch.cuda.memory_allocated()}')
            logging.info(f'max allocated: {torch.cuda.max_memory_allocated()}')
            logging.info(f'reserved: {torch.cuda.memory_reserved()}')

            model.eval()

            # part val first, for fairness of all methods
            start_time = time.time()
            if successful_part:
                loader, next_loader = next_loader, None

                while True:
                    data = loader.next()
                    if data is None:
                        loader = None
                        loss = None
                        corrects = None
                        break
                    else:
                        if data[1]:  # stop signal
                            successful_ppr = dataset.set_split('val_ppr')
                            if successful_ppr:
                                next_loader = get_prefetch_generator(self.mode,
                                                                     self.neighbor_sampling,
                                                                     dataset,
                                                                     val_nodes,
                                                                     batch_size=self.batch_size)
                            else:
                                dataset.set_split('val_self')
                                next_loader = get_prefetch_generator(self.mode,
                                                                     self.neighbor_sampling,
                                                                     dataset,
                                                                     val_nodes,
                                                                     batch_size=self.batch_size)

                    with torch.no_grad():
                        loss, corrects, num_nodes, _, _ = run_batch(model, MyGraph(*data[0]), verbose=False)
                        data_dic['part']['loss'] += loss
                        data_dic['part']['acc'] += corrects
                        data_dic['part']['num'] += num_nodes

            part_val_time = time.time() - start_time

            # ppr val
            start_time = time.time()
            if successful_ppr:
                loader, next_loader = next_loader, None

                while True:
                    data = loader.next()
                    if data is None:
                        loader = None
                        loss = None
                        corrects = None
                        break
                    else:
                        if data[1]:  # stop signal
                            dataset.set_split('val_self')
                            next_loader = get_prefetch_generator(self.mode,
                                                                 self.neighbor_sampling,
                                                                 dataset,
                                                                 val_nodes,
                                                                 batch_size=self.batch_size)

                    with torch.no_grad():
                        loss, corrects, num_nodes, _, _ = run_batch(model, MyGraph(*data[0]), verbose=False)
                        data_dic['ppr']['loss'] += loss
                        data_dic['ppr']['acc'] += corrects
                        data_dic['ppr']['num'] += num_nodes

            ppr_val_time = time.time() - start_time

            # original val
            loader, next_loader = next_loader, None
            start_time = time.time()

            while True:
                data = loader.next()
                if data is None:
                    loader = None
                    loss = None
                    corrects = None
                    break
                else:
                    if data[1]:  # stop signal
                        if epoch < self.epoch_max - 1:
                            dataset.set_split('train')
                            next_loader = get_prefetch_generator(self.mode,
                                                                 self.neighbor_sampling,
                                                                 dataset,
                                                                 train_nodes,
                                                                 self.batch_size)
                        else:
                            next_loader = None

                with torch.no_grad():
                    loss, corrects, num_nodes, _, _ = run_batch(model, MyGraph(*data[0]), verbose=False)
                    data_dic['self']['loss'] += loss
                    data_dic['self']['acc'] += corrects
                    data_dic['self']['num'] += num_nodes

            self_val_time = time.time() - start_time

            # update training info
            for cat in ['train', 'self', 'part', 'ppr']:
                if data_dic[cat]['num'] > 0:
                    data_dic[cat]['loss'] = (data_dic[cat]['loss'] / data_dic[cat]['num']).item()
                    data_dic[cat]['acc'] = (data_dic[cat]['acc'] / data_dic[cat]['num']).item()
                best_accs[cat] = max(best_accs[cat], data_dic[cat]['acc'])

            # lr scheduler
            criterion_val_loss = data_dic['part']['loss'] if data_dic['part']['loss'] != 0 else data_dic['self']['loss']
            if scheduler is not None:
                scheduler.step(criterion_val_loss)

            # early stop
            criterion_val_acc = data_dic['part']['acc'] if data_dic['part']['acc'] != 0 else data_dic['self']['acc']
            if criterion_val_acc > best_val_acc:
                best_val_acc = criterion_val_acc
                patience_count = 0
                torch.save(model.state_dict(), model_path)
            else:
                patience_count += 1
                if epoch > self.epoch_min and patience_count > self.patience:
                    scheduler = None
                    opt = None
                    assert loader is None

                    if next_loader is not None:
                        next_loader.stop_signal = True
                        while next_loader.isAlive():
                            batch = next_loader.next()
                        next_loader = None
                    torch.cuda.empty_cache()
                    break

            # set tqdm
            if self.notebook:
                pbar.set_postfix(train_loss='{:.3f}'.format(data_dic['train']['loss']),
                                 self_val_loss='{:.3f}'.format(data_dic['self']['loss']),
                                 part_val_loss='{:.3f}'.format(data_dic['part']['loss']),
                                 ppr_val_loss='{:.3f}'.format(data_dic['ppr']['loss']),
                                 train_acc='{:.3f}'.format(data_dic['train']['acc']),
                                 self_val_acc='{:.3f}'.format(data_dic['self']['acc']),
                                 part_val_acc='{:.3f}'.format(data_dic['part']['acc']),
                                 ppr_val_acc='{:.3f}'.format(data_dic['ppr']['acc']),
                                 lr='{:.5f}'.format(opt.param_groups[0]['lr']),
                                 patience='{:d} / {:d}'.format(patience_count, self.patience))

            # maintain curves
            training_curve['per_train_time'].append(train_time)
            training_curve['per_self_val_time'].append(self_val_time)
            training_curve['per_part_val_time'].append(part_val_time)
            training_curve['per_ppr_val_time'].append(ppr_val_time)
            training_curve['train_loss'].append(data_dic['train']['loss'])
            training_curve['train_acc'].append(data_dic['train']['acc'])
            training_curve['self_val_loss'].append(data_dic['self']['loss'])
            training_curve['self_val_acc'].append(data_dic['self']['acc'])
            training_curve['ppr_val_loss'].append(data_dic['ppr']['loss'])
            training_curve['ppr_val_acc'].append(data_dic['ppr']['acc'])
            training_curve['part_val_loss'].append(data_dic['part']['loss'])
            training_curve['part_val_acc'].append(data_dic['part']['acc'])
            training_curve['lr'].append(opt.param_groups[0]['lr'])

        #             writer.add_scalar('train_loss', data_dic['train']['loss'], epoch)
        #             writer.add_scalar('train_acc', data_dic['train']['acc'], epoch)
        #             writer.add_scalar('self_val_loss', data_dic['self']['loss'], epoch)
        #             writer.add_scalar('self_val_acc', data_dic['self']['acc'], epoch)

        # ending
        self.database['best_train_accs'].append(best_accs['train'])
        self.database['training_curves'].append(training_curve)

        logging.info(f"best train_acc: {best_accs['train']}, "
                     f"best self val_acc: {best_accs['self']}, "
                     f"best part val_acc: {best_accs['part']}"
                     f"best ppr val_acc: {best_accs['ppr']}")

        torch.cuda.empty_cache()
        assert next_loader is None and loader is None

    #         writer.flush()

    def train_single_batch(self,
                           dataset,
                           model,
                           lr,
                           reg,
                           val_per_epoch=5,
                           comment='',
                           run_no=''):
        raise NotImplementedError

    @torch.no_grad()
    def inference(self,
                  dataset,
                  model,
                  val_nodes=None,
                  test_nodes=None,
                  adj=None,
                  x=None,
                  y=None,
                  file_dir='/nfs/students/qian',
                  comment='',
                  run_no='',
                  full_infer=True,
                  clear_cache=False,
                  record_numbatch=False):

        model_dir = os.path.join(file_dir, comment)
        assert os.path.isdir(model_dir)
        model_path = os.path.join(model_dir, f'model_{run_no}.pt')

        model.load_state_dict(torch.load(model_path))
        model.eval()

        cat_dict = {('self', 'val',): [self.database['self_val_accs'], self.database['self_val_f1s']],
                    ('part', 'val',): [self.database['part_val_accs'], self.database['part_val_f1s']],
                    ('ppr', 'val',): [self.database['ppr_val_accs'], self.database['ppr_val_f1s']],
                    ('self', 'test',): [self.database['self_test_accs'], self.database['self_test_f1s']],
                    ('part', 'test',): [self.database['part_test_accs'], self.database['part_test_f1s']],
                    ('ppr', 'test',): [self.database['ppr_test_accs'], self.database['ppr_test_f1s']], }

        data_dict = {'val': val_nodes, 'test': test_nodes}

        time_dict = {'self': self.database['self_inference_time'],
                     'part': self.database['part_inference_time'],
                     'ppr': self.database['ppr_inference_time']}

        # redundant run to warm up
        #         for cat in ['val', 'test']:
        #             for sample in ['self', 'LBMB']:
        #                 success_ful = dataset.set_split(cat + '_' + sample)
        #                 if success_ful:
        #                     loader = get_prefetch_generator(self.mode,
        #                                                     self.neighbor_sampling,
        #                                                     dataset,
        #                                                     test_nodes,
        #                                                     batch_size=self.batch_size)

        #                     while True:
        #                         data = loader.next()
        #                         if data is None:
        #                             del loader
        #                             break

        #                         _, _, _, pred_label_batch, true_label_batch = run_batch(model, MyGraph(*data[0]), verbose=True)

        for cat in ['val', 'test']:
            for sample in ['self', 'part', 'ppr']:
                acc, f1 = 0., 0.
                num_batch = 0
                torch.cuda.synchronize()
                start_time = time.time()
                success_ful = dataset.set_split(cat + '_' + sample)
                if success_ful:
                    loader = get_prefetch_generator(self.mode,
                                                    self.neighbor_sampling,
                                                    dataset,
                                                    test_nodes,
                                                    batch_size=self.batch_size)

                    pred_labels = []
                    true_labels = []

                    while True:
                        data = loader.next()
                        if data is None:
                            del loader
                            break

                        _, _, _, pred_label_batch, true_label_batch = run_batch(model, MyGraph(*data[0]), verbose=True)
                        pred_labels.append(pred_label_batch)
                        true_labels.append(true_label_batch)
                        num_batch += 1

                    #                         print(cat, torch.cuda.max_memory_allocated())

                    pred_labels = np.concatenate(pred_labels, axis=0)
                    true_labels = np.concatenate(true_labels, axis=0)

                    acc = (pred_labels == true_labels).sum() / len(true_labels)
                    f1 = f1_score(true_labels, pred_labels, average='macro', zero_division=0)

                cat_dict[(sample, cat,)][0].append(acc)
                cat_dict[(sample, cat,)][1].append(f1)

                if record_numbatch:
                    self.database[f'numbatch_{sample}_{cat}'].append(num_batch)

                logging.info("{}_{}_acc: {:.3f}, {}_{}_f1: {:.3f}, ".format(sample, cat, acc, sample, cat, f1))
                if cat == 'test':
                    torch.cuda.synchronize()
                    time_dict[sample].append(time.time() - start_time)

                if clear_cache:
                    dataset.clear_cur_cache()

        # chunked full-batch inference
        if full_infer:
            start_time = time.time()

            mask = np.union1d(val_nodes, test_nodes)
            val_mask = np.in1d(mask, val_nodes)
            test_mask = np.in1d(mask, test_nodes)
            assert np.all(np.invert(val_mask) == test_mask)
            #             num_chunks = max(len(dataset.train_loader), len(dataset.val_loader[0]), len(dataset.test_loader[0]))
            outputs = model.chunked_pass(MyGraph(x=x, adj=adj, idx=torch.from_numpy(mask)),
                                         self.num_batches[0] // self.batch_size).numpy()

            for cat in ['val', 'test']:
                nodes = val_nodes if cat == 'val' else test_nodes
                _mask = val_mask if cat == 'val' else test_mask
                pred = np.argmax(outputs[_mask], axis=1)
                true = y.detach().numpy()[nodes]

                acc = (pred == true).sum() / len(true)
                f1 = f1_score(true, pred, average='macro', zero_division=0)

                self.database[f'full_{cat}_accs'].append(acc)
                self.database[f'full_{cat}_f1s'].append(f1)

                logging.info("full_{}_acc: {:.3f}, full_{}_f1: {:.3f}, ".format(cat, acc, cat, f1))

            self.database['full_inference_time'].append(time.time() - start_time)
