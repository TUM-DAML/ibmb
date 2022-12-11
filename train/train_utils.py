import torch


def run_batch(model, graph, num_microbatches_minibatch=None):
    y = graph.y[graph.output_node_mask]
    num_prime_nodes = len(y)
    outputs = model(graph)

    if hasattr(graph, 'node_norm') and graph.node_norm is not None:
        loss = torch.nn.functional.nll_loss(outputs, y, reduction='none')
        loss = (loss * graph.node_norm).sum()
    else:
        loss = torch.nn.functional.nll_loss(outputs, y)
        
    return_loss = loss.clone().detach() * num_prime_nodes
    
    if model.training:
        loss = loss / num_microbatches_minibatch
        loss.backward()
    
    pred = torch.argmax(outputs, dim=1)
    corrects = pred.eq(y).sum().detach()

    return return_loss, corrects, num_prime_nodes, pred, y
