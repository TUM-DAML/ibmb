def get_ppr_default(dataset: str, model: str):
    ppr_default = {
        'sage':
            {'arxiv':
                 {'neighbor_topk': 16,
                  'merge_max_size': None,
                  'primes_per_batch': 9000,
                  'pushflowthresh': 2e-4, },
             'products':
                 {'neighbor_topk': 64,
                  'merge_max_size': None,
                  'primes_per_batch': 4500,
                  'pushflowthresh': 5e-4, },
             'reddit':
                 {'neighbor_topk': 8,
                  'merge_max_size': [80000, 160000, 160000],
                  'primes_per_batch': 35000,
                  'pushflowthresh': 2e-5, },
             },
        'gcn':
            {'arxiv':
                 {'neighbor_topk': 16,
                  'merge_max_size': None,
                  'primes_per_batch': 9000,
                  'pushflowthresh': 2e-4, },
             'products':
                 {'neighbor_topk': 64,
                  'merge_max_size': None,
                  'primes_per_batch': 4500,
                  'pushflowthresh': 5e-4, },
             'reddit':
                 {'neighbor_topk': 8,
                  'merge_max_size': [80000, 160000, 160000],
                  'primes_per_batch': 35000,
                  'pushflowthresh': 2e-5, },
             },
        'gat':
            {'arxiv':
                 {'neighbor_topk': 16,
                  'merge_max_size': None,
                  'primes_per_batch': 4000,
                  'pushflowthresh': 2e-4, },
             'products':
                 {'neighbor_topk': 64,
                  'merge_max_size': None,
                  'primes_per_batch': 130,
                  'pushflowthresh': 5e-4, },
             'reddit':
                 {'neighbor_topk': 8,
                  'merge_max_size': [10000, 20000, 20000],
                  'primes_per_batch': 2500,
                  'pushflowthresh': 2e-5, },
             },
    }
    return ppr_default[model][dataset]


ppr_iter_prams = {'arxiv': {'chunksize': 10000,
                   'alpha': 0.05,
                   'iters': 3,
                   'top_percent': [1, 0.01, 0.25],
                   'thresh': 0.001},
         'products': {'chunksize': 10000,
                   'alpha': 0.05,
                   'iters': 3,
                   'top_percent': [1, 0.01, 0.15],
                   'thresh': 0.001},
         'reddit': {'chunksize': 8000,
                   'alpha': 0.5,
                   'iters': 3,
                   'top_percent': [0.1, 0.01, 0.15],
                   'thresh': 0.0001},
                  'papers100M': {'chunksize': 8000,
                   'alpha': 0.25,
                   'iters': 3,
                   'top_percent': [0.1, 0.01, 0.15],
                   'thresh': 1e-4}, }
