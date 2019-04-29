"""
Remote file for multi-shot KMeans
"""

import os
import sys
import json
import logging
import configparser
import numpy as np
import utils as ut
from . import remote_computations as remote
from . import local_computations as local

CONFIG_FILE = 'dkm_config.cfg'
DEFAULT_k = 5
DEFAULT_epsilon = 0.00001
DEFAULT_shuffle = True
DEFAULT_learning_rate = 0.001
DEFAULT_verbose = True
DEFAULT_optimization = 'lloyd'


def dkm_remote_stop(**kwargs):
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
    computation_output = dict(output=dict(computation_phase="dkm_remote_stop"),
                              )
    return computation_output


def dkm_remote_init_env(args,
                        config_file=CONFIG_FILE,
                        k=DEFAULT_k,
                        optimization=DEFAULT_optimization,
                        epsilon=DEFAULT_epsilon,
                        learning_rate=DEFAULT_learning_rate,
                        verbose=DEFAULT_verbose):
    """
        # Description:
            Initialize the remote environment, creating the config file.

        # PREVIOUS PHASE:
            None

        # INPUT:

            |   name            |   type    |   default     |
            |   ---             |   ---     |   ---         |
            |   config_file     |   str     |   config.cfg  |
            |   k               |   int     |   5           |
            |   optimization    |   str     |   lloyd       |
            |   epsilon         |   float   |   0.00001     |
            |   shuffle         |   bool    |   False       |
            |   data_file       |   str     |   data.txt    |
            |   learning_rate   |   float   |   0.001       |
            |   verbose         |   float   |   True        |

        # OUTPUT:
            - config file written to disk
            - k
            - learning_rate
            - optimization
            - shuffle

        # NEXT PHASE:
            local_init_env
    """
    state = args['state']
    inputs = args['input']
    cache = args['cache']
    ut.log('REMOTE: Initializing remote environment', state)
    config_path = os.path.join(state['outputDirectory'], config_file)
    if not os.path.exists(config_path):
        config = configparser.ConfigParser()
        config['REMOTE'] = dict(k=k,
                                optimization=optimization,
                                epsilon=epsilon,
                                learning_rate=learning_rate,
                                verbose=verbose)
        with open(config_path, 'w') as file:
            config.write(file)
    # output
    computation_output = dict(
        output=dict(work_dir='/computation',
                    config_file=config_path,
                    k=k,
                    learning_rate=learning_rate,
                    optimization=optimization,
                    shuffle=True,
                    computation_phase="dkm_remote_init_env"),
        state=state,
        cache=cache
    )
    return computation_output


def dkm_remote_init_centroids(args, config_file=CONFIG_FILE, **kwargs):
    """
        # Description:
            Initialize K centroids from locally selected centroids.

        # PREVIOUS PHASE:
            local_init_centroids

        # INPUT:

            |   name             |   type    |   default     |
            |   ---              |   ---     |   ---         |
            |   config_file      |   str     |   config.cfg  |

        # OUTPUT:
            - centroids: list of numpy arrays

        # NEXT PHASE:
            local_compute_optimizer
    """
    state = args['state']
    inputs = args['input']
    ut.log('REMOTE: Initializing centroids', state)
    config_file = os.path.join(state['outputDirectory'], config_file)
    config = configparser.ConfigParser()
    config.read(config_file)
    k = config['k']
    # Have each site compute k initial clusters locally
    local_centroids = [cent for site in args for cent in args[site]]
    # and select k random clusters from the s*k pool
    np.random.shuffle(local_centroids)
    remote_centroids = local_centroids[:k]
    computation_output = dict(
        output=dict(
            work_dir='.',
            config_file=config_file,
            # local_centroids=remote_centroids,
            computation_phase="dkm_remote_init_centroids",
            remote_centroids=remote_centroids
        ),
        state=state
    )
    return computation_output


def dkm_remote_aggregate_optimizer(args, config_file=CONFIG_FILE):
    """
        # Description:
            Aggregate optimizers sent from local nodes.

        # PREVIOUS PHASE:
            local_compute_optimizer

        # INPUT:

            |   name             |   type    |   default     |
            |   ---              |   ---     |   ---         |
            |   config_file      |   str     |   config.cfg  |

        # OUTPUT:
            - remote_optimizer: list of K numpy arrays

        # NEXT PHASE:
            remote_optimization_step
    """
    state = args['state']
    inputs = args['input']
    config_file = os.path.join(state['outputDirectory'], config_file)
    config = configparser.ConfigParser()
    config.read(config_file)

    optimization = config['REMOTE']['optimization']
    ut.log('REMOTE: Aggregate optimizer', state)
    local_optimizers = [site['local_optimizer'] for site in args]
    s = len(local_optimizers)
    remote_optimizer = remote.aggregate_sum(local_optimizers)
    if optimization == 'lloyd':
        # for the mean, we need to further divide the number of sites
        remote_optimizer = [r / s for r in remote_optimizer]

    computation_output = dict(
        output=dict(
            remote_optimizer=remote_optimizer,
            computation_phase="dkm_remote_aggregate_optimizer"
        ),
        state=state
    )
    return computation_output


def dkm_remote_optimization_step(args, config_file=CONFIG_FILE):
    """
        # Description:
            Use optimizer to take the next step.

        # PREVIOUS PHASE:
            remote_aggregate_optimizer

        # INPUT:

            |   name               |   type    |   default     |
            |   ---                |   ---     |   ---         |
            |   config_file        |   str     |   config.cfg  |
            |   remote_centroids   |   list    |   None        |
            |   remote_optimizer   |   list    |   None        |

        # OUTPUT:
            - previous centroids: list of numpy arrays
            - remote centroids: list of numpy arrays

        # NEXT PHASE:
            remote_check_convergence
    """
    state = args['state']
    inputs = args['input']
    config_path = os.path.join(state['outputDirectory'], config_file)
    remote_centroids = inputs['remote_centroids']
    remote_optimizer = inputs['remote_optimizer']
    ut.log('REMOTE: Optimization step', args['state'])
    config = configparser.ConfigParser()
    config.read(config_file)
    optimization = config['REMOTE']['optimization']
    if optimization == 'lloyd':
        # Then, update centroids as corresponding to the local mean
        previous_centroids = remote_centroids[:]
        remote_centroids = remote_optimizer[:]
    elif optimization == 'gradient':
        # Then, update centroids according to one step of gradient descent
        [remote_centroids, previous_centroids] = \
            local.gradient_step(remote_optimizer, remote_centroids)
    computation_output = dict(output=dict(
        computation_phase="dkm_remote_optimization_step",
        previous_centroids=previous_centroids,
        remote_centroids=remote_centroids),
        state=state
    )
    return computation_output


def dkm_remote_check_convergence(args):
    """
        # Description:
            Check convergence.

        # PREVIOUS PHASE:
            remote_aggregate_optimizer

        # INPUT:

            |   name               |   type    |   default     |
            |   ---                |   ---     |   ---         |
            |   config_file        |   str     |   config.cfg  |
            |   remote_centroids   |   list    |   None        |
            |   previous_centroids |   list    |   None        |

        # OUTPUT:
            - boolean encoded in name of phase
            - delta
            - remote_centroids

        # NEXT PHASE:
            remote_check_convergence
    """
    state = args['state']
    inputs = args['input']
    config_file = inputs['config_file']
    remote_centroids = inputs['remote_centroids']
    previous_centroids = inputs['previous_centroids']
    ut.log('REMOTE: Check convergence', state)
    config = configparser.ConfigParser()
    config.read(config_file)
    epsilon = config['REMOTE']['epsilon']
    remote_check, delta = local.check_stopping(remote_centroids,
                                               previous_centroids, epsilon)
    new_phase = "dkm_remote_converged_true" if remote_check else "dkm_remote_converged_false"
    computation_output = dict(output=dict(
        computation_phase=new_phase,
        delta=delta,
        remote_centroids=remote_centroids,
    ),
        state=state
    )
    return computation_output


def dkm_remote_aggregate_output(args):
    """
        # Description:
            Check convergence.

        # PREVIOUS PHASE:
            remote_check_convergence

        # INPUT:

            |   name               |   type    |   default     |
            |   ---                |   ---     |   ---         |
            |   config_file        |   str     |   config.cfg  |
            |   remote_centroids   |   list    |   None        |
            |   previous_centroids |   list    |   None        |

        # OUTPUT:
            -remote_centroids

    """
    state = args['state']
    inputs = args['input']
    remote_centroids = inputs['remote_centroids']
    ut.log('REMOTE: Aggregating input', state)
    computation_output = dict(output=dict(
        computation_phase="dkm_remote_aggregate_output",
        remote_centroids=remote_centroids,
    ),
        state=state
    )
    return computation_output


if __name__ == '__main__':

    parsed_args = json.loads(sys.stdin.read())
    phase_key = list(listRecursive(parsed_args, 'computation_phase'))

    if 'local_noop' in phase_key:  # FIRST PHASE
        computation_output = remote_init_env(parsed_args['input'])
        sys.stdout.write(computation_output)
    elif 'local_init_centroids' in phase_key:  # LOCAL -> REMOTE
        computation_output = remote_init_centroids(parsed_args['input'])
        sys.stdout.write(computation_output)
    elif 'local_compute_optimizer' in phase_key:  # LOCAL -> REMOTE
        computation_output = remote_aggregate_optimizer(parsed_args['input'])
        computation_output = remote_optimization_step(**computation_output)
        sys.stdout.write(computation_output)
    elif 'local_compute_clustering_2' in phase_key:  # LOCAL -> REMOTE
        computation_output = remote_check_convergence(parsed_args['input'])
        if 'remote_converged_true' in computation_output['output'][
                'computation_phase']:
            computation_output = remote_aggregate_output(**computation_output)
        sys.stdout.write(computation_output)
    else:
        raise ValueError('Oops')
