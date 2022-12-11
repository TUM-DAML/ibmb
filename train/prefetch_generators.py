import queue
import threading

from dataloaders.GraphSAINTRWSampler import SaintRWValSampler
from dataloaders.IBMBRandLoader import IBMBRandLoader
from dataloaders.ShaDowLoader import ShaDowLoader
from dataloaders.LADIESSampler import LADIESSampler
from dataloaders.NeighborSamplingLoader import NeighborSamplingLoader


class BaseGenerator(threading.Thread):
    def __init__(self, max_prefetch=1, device='cuda'):
        threading.Thread.__init__(self)
        self.queue = queue.Queue(max_prefetch)
        self.device = device
        self.stop_signal = False
        self.start()

    def run(self):
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class BackgroundGenerator(BaseGenerator):
    def __init__(self, dataloader, max_prefetch=1, device='cuda'):
        self.dataloader = dataloader
        super().__init__(max_prefetch, device)

    def run(self):
        for i, graph in enumerate(self.dataloader):
            if isinstance(self.dataloader, (SaintRWValSampler,
                                            ShaDowLoader,
                                            IBMBRandLoader,
                                            LADIESSampler,
                                            NeighborSamplingLoader)):
                stop_signal = i == self.dataloader.loader_len() - 1
            else:
                stop_signal = i == len(self.dataloader) - 1
            self.queue.put((graph.to(self.device, non_blocking=True), stop_signal))
        self.queue.put(None)
