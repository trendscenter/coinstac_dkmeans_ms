import os
import sys
import logging
import numpy as np
import utils as ut
import configparser
from . import local_computations as local

CONFIG_FILE = 'dkm_config.cfg'
DEFAULT_data_file = 'data.txt'
DEFAULT_k = 5
DEFAULT_shuffle = True
DEFAULT_learning_rate = 0.001
DEFAULT_optimization = 'lloyd'


def dkm_local_noop(args, **kwargs):
    """
        # Description:
            Nooperation

        # PREVIOUS PHASE:
            NA

        # INPUT:

        # OUTPUT:

        # NEXT PHASE:
            remote_init_env
    """
    computation_output = dict(output=dict(computation_phase="dkm_local_noop"),
                              )
    return computation_output


def dkm_local_init_env(args,
                       config_file=CONFIG_FILE,
                       k=DEFAULT_k,
                       optimization=DEFAULT_optimization,
                       shuffle=DEFAULT_shuffle,
                       learning_rate=DEFAULT_learning_rate,
                       **kwargs):
    """
        # Description:
            Initialize the local environment, creating the config file.

        # PREVIOUS PHASE:
            remote_init_env

        # INPUT:

            |   name            |   type    |   default     |
            |   ---             |   ---     |   ---         |
            |   config_file     |   str     |   config.cfg  |
            |   k               |   int     |   5           |
            |   optimization    |   str     |   lloyd       |
            |   shuffle         |   bool    |   False       |
            |   data_file       |   str     |   data.txt    |
            |   learning_rate   |   float   |   0.001       |

        # OUTPUT:
            - config file written to disk

        # NEXT PHASE:
            local_init_centroids
    """
    state, inputs, cache = ut.resolve_args(args)
    data_file = ut.resolve_input('all_windows', cache)
    ut.log('LOCAL: Initializing remote environment', state)
    config_path = os.path.join(state['outputDirectory'], config_file)
    cache['config_file'] = config_path
    config = configparser.ConfigParser()
    config['LOCAL'] = dict(k=k,
                           optimization=optimization,
                           shuffle=shuffle,
                           data_file=data_file,
                           learning_rate=learning_rate)
    with open(config_path, 'w') as file:
        config.write(file)
    # output
    computation_output = dict(
        output=dict(
            config_file=config_path,
            computation_phase="dkm_local_init_env"),
        state=state,
        cache=cache
    )
    return computation_output


def dkm_local_init_centroids(args,
                             config_file=CONFIG_FILE,
                             **kwargs):
    """
        # Description:
            Initialize K centroids from own data.

        # PREVIOUS PHASE:
            local_init_env

        # INPUT:

            |   name             |   type    |   default     |
            |   ---              |   ---     |   ---         |
            |   config_file      |   str     |   config.cfg  |

        # OUTPUT:
            - centroids: list of numpy arrays

        # NEXT PHASE:
            remote_init_centroids
    """
    state, inputs, cache = ut.resolve_args(args)
    config_file = ut.resolve_input('config_file', cache)
    ut.log('LOCAL: Initializing centroids', state)
    config = configparser.ConfigParser()
    config.read(config_file)
    data = np.load(config['LOCAL']['data_file'])
    centroids = local.initialize_own_centroids(data, int(config['LOCAL']['k']))
    np.save(os.path.join(state['outputDirectory'], 'initial_centroids'), 'centroids')
    ut.log('Local centroids looks like %s' % (str(type(centroids))), state)
    # output
    cache['local_centroids'] = centroids
    computation_output = dict(output=dict(
        config_file=config_file,
        local_centroids=centroids,
        computation_phase="dkm_local_init_env"),
        state=state,
        cache=cache
    )
    return computation_output


def dkm_local_compute_clustering(args,
                                 config_file=CONFIG_FILE,
                                 **kwargs):
    """
        # Description:
            Assign data instances to clusters.

        # PREVIOUS PHASE:
            remote_init_centroids (on first run only)
            remote_cehck_convergence

        # INPUT:

            |   name                |   type    |   default     |
            |   ---                 |   ---     |   ---         |
            |   config_file         |   str     |   config.cfg  |
            |   remote_centroids    |   list    |   None        |
            |   computation_phase   |   list    |   None        |

        # OUTPUT:
            - centroids: list of numpy arrays

        # NEXT PHASE:
            remote_init_centroids
    """
    state, inputs, cache = ut.resolve_args(args)
    config_file = ut.resolve_input('config_file', cache)
    remote_centroids = ut.resolve_input('remote_centroids', inputs)
    computation_phase = ut.resolve_input('computation_phase', inputs)
    ut.log('LOCAL: computing clustering', state)
    if remote_centroids is None:
        raise ValueError(
            "LOCAL: at local_compute_clustering - remote_centroids not passed correctly"
        )
    if computation_phase is None:
        raise ValueError(
            "LOCAL: at local_compute_clustering - computation_phase not passed correctly"
        )
    config = configparser.ConfigParser()
    config.read(config_file)
    ut.log('Config file is %s, with keys %s' % (config_file, str(dict(config))), state)

    data = np.load(config['LOCAL']['data_file'])

    cluster_labels = local.compute_clustering(data, remote_centroids)

    new_comp_phase = "dkm_local_compute_clustering"
    if computation_phase == "dkm_remote_optimization_step":
        new_comp_phase = "dkm_local_compute_clustering_2"

    computation_output = ut.default_computation_output(args)
    cache['cluster_labels'] = cluster_labels
    cache['remote_centroids'] = remote_centroids
    computation_output['output'] = dict(
        computation_phase=new_comp_phase,
        remote_centroids=remote_centroids,
        cluster_labels=cluster_labels
    )
    computation_output['cache'] = cache
    return computation_output


def dkm_local_compute_optimizer(args,
                                config_file=CONFIG_FILE,
                                **kwargs):
    """
        # Description:
            Compute local optimizers with local data.

        # PREVIOUS PHASE:
            local_compute_clustering

        # INPUT:

            |   name                |   type    |   default     |
            |   ---                 |   ---     |   ---         |
            |   config_file         |   str     |   config.cfg  |
            |   remote_centroids    |   list    |   None        |
            |   cluster_labels      |   list    |   None        |

        # OUTPUT:
            - centroids: list of numpy arrays

        # NEXT PHASE:
            remote_init_centroids
    """
    state, inputs, cache = ut.resolve_args(args)
    config_file = ut.resolve_input('config_file', cache)
    remote_centroids = ut.resolve_input('remote_centroids', inputs, cache)
    cluster_labels = ut.resolve_input('cluster_labels', inputs, cache)
    if remote_centroids is None:
        raise ValueError(
            "LOCAL: at local_compute_clustering - remote_centroids not passed correctly"
        )
    if cluster_labels is None:
        raise ValueError(
            "LOCAL: at local_compute_clustering - cluster_labels not passed correctly"
        )
    ut.log('LOCAL: computing optimizers', state)
    config = configparser.ConfigParser()
    config.read(config_file)
    data = np.load(config['LOCAL']['data_file'])
    k = int(config['LOCAL']['k'])
    learning_rate = config['LOCAL']['learning_rate']
    optimization = config['LOCAL']['optimization']
    if optimization == 'lloyd':
        local_optimizer = local.compute_mean(data, cluster_labels, k)
    elif optimization == 'gradient':
        # Gradient descent has sites compute gradients locally
        local_optimizer = local.compute_gradient(data, cluster_labels[i], remote_centroids, learning_rate)

    outdir = state['outputDirectory']
    np.save(os.path.join(outdir, 'local_optimizer.npy'), local_optimizer)
    np.save(os.path.join(outdir, 'local_cluster_labels.npy'), cluster_labels)

    """ Debugged by AK """
    local_optimizer = [l.tolist() if isinstance(l, np.ndarray) else l for l in local_optimizer]
    computation_output = dict(output=dict(
        local_optimizer=local_optimizer,
        computation_phase="dkm_local_compute_optimizer"),
        state=state
    )
    return computation_output
