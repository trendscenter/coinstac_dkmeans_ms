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


def dkm_remote_stop(args, **kwargs):
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
    state, inputs, cache = ut.resolve_args(args)
    computation_output = dict(
        output=dict(computation_phase="dkm_remote_stop"),
        state=state,
        cache=cache
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
    state, inputs, cache = ut.resolve_args(args)
    ut.log('REMOTE: Initializing remote environment', state)
    config_path = os.path.join(state['outputDirectory'], config_file)
    config = configparser.ConfigParser()
    config['REMOTE'] = dict(k=k,
                            optimization=optimization,
                            epsilon=epsilon,
                            learning_rate=learning_rate,
                            verbose=verbose)
    with open(config_path, 'w') as file:
        config.write(file)
    # output
    cache['config_file'] = config_path
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
    state, inputs, cache = ut.resolve_args(args)
    ut.log('REMOTE: Initializing centroids', state)
    config_file = ut.resolve_input('config_file', cache)
    config = configparser.ConfigParser()
    config.read(config_file)
    ut.log('Config file %s, looks like %s' % (config_file, str(dict(config))), state)
    k = int(config['REMOTE']['k'])
    # Have each site compute k initial clusters locally
    local_centroids = []
    if 'remote_centroids' in inputs.keys():
        remote_centroids = inputs['remote_centroids']
    elif 'remote_centroids' in cache.keys():
        remote_centroids = cache['remote_centroids']
    else:
        for site in inputs:
            ut.log('Local site %s sent inputs with keys %s' % (site, str(inputs[site].keys())), state)
            local_centroids += inputs[site]['local_centroids']
        # and select k random clusters from the s*k pool
        np.random.shuffle(local_centroids)
        remote_centroids = local_centroids[:k]
    cache['config_file'] = config_file
    cache['remote_centroids'] = remote_centroids
    computation_output = dict(
        output=dict(
            work_dir='.',
            config_file=config_file,
            # local_centroids=remote_centroids,
            computation_phase="dkm_remote_init_centroids",
            remote_centroids=remote_centroids
        ),
        state=state,
        cache=cache
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
    state, inputs, cache = ut.resolve_args(args)
    config_file = ut.resolve_input('config_file', cache)
    config = configparser.ConfigParser()
    config.read(config_file)

    optimization = config['REMOTE']['optimization']
    ut.log('REMOTE: Aggregate optimizer', state)
    local_optimizers = [inputs[site]['local_optimizer'] for site in inputs]
    s = len(local_optimizers)
    remote_optimizer = remote.aggregate_sum(local_optimizers)
    if not all([type(r) is np.ndarray for r in remote_optimizer]):
        try:
            remote_opt2 = [np.array(c) for c in remote_optimizer]
            remote_optimizer = remote_opt2[:]
        except Exception as e:
            raise (Exception("Hit valueerror. Remote optimizer types are %s" % ([len(r) for r in remote_optimizer])))
    if optimization == 'lloyd':
        # for the mean, we need to further divide the number of sites
        try:
            remote_optimizer = [r / s for r in remote_optimizer]
        except Exception as e:
            raise (Exception("Hit valueerror. Remote optimizer types are %s" % ([len(r) for r in remote_optimizer])))

    """Debugged by AK"""
    remote_optimizer = [a.tolist() if isinstance(a, np.ndarray) else a for a in remote_optimizer]
    cache['remote_optimizer'] = remote_optimizer
    computation_output = dict(
        output=dict(
            remote_optimizer=remote_optimizer,
            computation_phase="dkm_remote_aggregate_optimizer"
        ),
        state=state,
        cache=cache
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
    state, inputs, cache = ut.resolve_args(args)
    config_file = ut.resolve_input('config_file', cache)
    remote_centroids = ut.resolve_input('remote_centroids', inputs, cache)
    remote_optimizer = ut.resolve_input('remote_optimizer', inputs, cache)
    if type(remote_centroids[0]) is not np.ndarray:
        remote_centroids = [np.array(c) for c in remote_centroids]
    ut.log('REMOTE: Optimization step', args['state'])
    config = configparser.ConfigParser()
    config.read(config_file)
    optimization = config['REMOTE']['optimization']
    if optimization == 'lloyd':
        # Then, update centroids as corresponding to the local mean
        previous_centroids = remote_centroids[:]
        remote_centroids = remote_optimizer[:]
        ut.log("Previous centroids look like %s" % type(previous_centroids[0]), state)
        ut.log("Remote centroids look like %s" % type(remote_centroids[0]), state)
    elif optimization == 'gradient':
        # Then, update centroids according to one step of gradient descent
        [remote_centroids, previous_centroids] = local.gradient_step(remote_optimizer, remote_centroids)

    """ Debugged by AK """
    previous_centroids = [a.tolist() if isinstance(a, np.ndarray) else a for a in previous_centroids]
    remote_centroids = [a.tolist() if isinstance(a, np.ndarray) else a for a in remote_centroids]

    cache['previous_centroids'] = previous_centroids
    cache['remote_centroids'] = remote_centroids
    computation_output = dict(output=dict(
        computation_phase="dkm_remote_optimization_step",
        remote_centroids=remote_centroids),
        state=state,
        cache=cache
    )
    return computation_output


def dkm_remote_check_convergence(args, config_file=CONFIG_FILE):
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
    state, inputs, cache = ut.resolve_args(args)
    ut.log('REMOTE: Check convergence', state)
    config_file = ut.resolve_input('config_file', cache)
    remote_centroids = ut.resolve_input('remote_centroids', inputs, cache)
    previous_centroids = ut.resolve_input('previous_centroids', inputs, cache)
    if type(remote_centroids) is not np.ndarray:
        remote_centroids = [np.array(c) for c in remote_centroids]
    if type(previous_centroids) is not np.ndarray:
        previous_centroids = [np.array(c) for c in previous_centroids]

    config = configparser.ConfigParser()
    config.read(config_file)
    epsilon = float(config['REMOTE']['epsilon'])
    remote_check, delta = local.check_stopping(remote_centroids,
                                               previous_centroids, epsilon)
    ut.log('REMOTE: Convergence Delta is %f, Converged is %s' % (delta, remote_check), state)
    new_phase = "dkm_remote_converged_true" if remote_check else "dkm_remote_converged_false"

    """Debugged by AK"""
    remote_centroids = [a.tolist() if isinstance(a, np.ndarray) else a for a in remote_centroids]
    computation_output = dict(
        output=dict(
            computation_phase=new_phase,
            delta=delta,
            remote_centroids=remote_centroids,
        ),
        state=state,
        cache=cache
    )
    return computation_output


def dkmnx_remote_check_convergence(args, config_file=CONFIG_FILE):
    computation_output = dkm_remote_check_convergence(args)
    new_phase = "dkmnx_remote_converged_true" if 'true' in computation_output[
        'output']['computation_phase'] else "dkmnx_remote_converged_false"
    computation_output['output']['computation_phase'] = new_phase
    return computation_output


def dkm_remote_aggregate_output(args):
    """
        # Description:
            Check convergence.

        # PREVIOUS PHASE:
            remote_check_convergence

        # INPUT:

            |   name | type | default |
            | --- | --- | --- |
            |   config_file | str | config.cfg |
            |   remote_centroids | list | None |
            |   previous_centroids | list | None |

        # OUTPUT:
            -remote_centroids

    """
    state, inputs, cache = ut.resolve_args(args)
    remote_centroids = ut.resolve_input('remote_centroids', inputs, cache)
    ut.log('REMOTE: Aggregating input', state)
    computation_output = dict(output=dict(
        computation_phase="dkm_remote_aggregate_output",
        remote_centroids=remote_centroids,
    ),
        state=state,
        cache=cache
    )
    return computation_output
