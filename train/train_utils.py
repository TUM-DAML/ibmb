import torch
from torch_sparse import SparseTensor


class MyGraph:
    def __init__(self, 
                 x: torch.tensor = None, 
                 y: torch.tensor = None, 
                 adj: [list, SparseTensor] = None, 
                 idx: torch.tensor = None, 
                 weight: torch.tensor = None):
        super().__init__()
        self.x = x
        self.y = y
        self.adj = adj
        self.idx = idx
        self.weight = weight
    
    def to(self, device):
        self.x = self.x.to(device) if self.x is not None else None
        self.y = self.y.to(device) if self.y is not None else None
        self.idx = self.idx.to(device) if self.idx is not None else None
        self.weight = self.weight.to(device) if self.weight is not None else None
        
        if self.adj is not None:
            if isinstance(self.adj, SparseTensor):
                self.adj = self.adj.to(device)
            else:
                self.adj = [adj.to(device) for adj in self.adj]


def run_batch(model, graph, num_microbatches_minibatch=None, verbose=False):
    num_prime_nodes = len(graph.y)
    outputs = model(graph)
#     logging.info(f'forward: {torch.cuda.memory_allocated()}')
    
    if graph.weight is None:
        loss = torch.nn.functional.nll_loss(outputs, graph.y)
    else:
        loss = torch.nn.functional.nll_loss(outputs, graph.y, reduction='none')
        loss = (loss * graph.weight).sum()
        
    return_loss = loss.clone().detach() * num_prime_nodes
    
    if model.training:
        loss = loss / num_microbatches_minibatch
        loss.backward()
    
    pred = torch.argmax(outputs, dim=1)
    corrects = pred.eq(graph.y).sum().detach()
    
    if verbose:
        pred = pred.cpu().detach().numpy()
        true = graph.y.cpu().detach().numpy()
    else:
        pred, true = None, None

    return return_loss, corrects, num_prime_nodes, pred, true
