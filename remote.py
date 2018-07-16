"""
Remote file for multi-shot KMeans
"""

import os
import sys
import json
import logging
import configparser
import numpy as np


DEFAULT_work_dir = './.coinstac_tmp/'
DEFAULT_config_file = 'config.cfg'
DEFAULT_k = 5
DEFAULT_epsilon = 0.00001
DEFAULT_shuffle = True
DEFAULT_lr = 0.001
DEFAULT_verbose = True
DEFAULT_optimization = 'lloyd'


def remote_init_env(work_dir=DEFAULT_work_dir, config_file=DEFAULT_config_file, k=DEFAULT_k,
                    optimization=DEFAULT_optimization, epsilon=DEFAULT_epsilon, lr=DEFAULT_lr,
                    verbose=DEFAULT_verbose):
    """
        Initialize the remote environment, config file if necessary.
    """
    logging.info('REMOTE: Initializing remote environment')
    # initialize environment
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    # create parameter / config file if it doesn't exist
    config_path = os.path.join(work_dir, config_file)
    if not os.path.exists(config_path):
        config = configparser.ConfigParser()
        config['REMOTE'] = dict(k=k, optimization=optimization, epsilon=epsilon,
                                lr=lr, verbose=verbose)
        with open(config_path, 'w') as file:
            config.write(file)
    # output
    computation_output = dict(output=
                              dict(
                                  work_dir=work_dir,
                                  config_file=config_file,
                                  k=k,
                                  optimization=optimization,
                                  shuffle=shuffle,
                                  computation_phase="remote_init_env"
                                  )
                              )
    return json.dumps(computation_output)


def remote_init_centroids(args, work_dir=DEFAULT_work_dir, config_file=DEFAULT_config_file):
    """
        Select K random centroids from the local centroids
    """
    logging.info('REMOTE: Initializing centroids')
    # Have each site compute k initial clusters locally
    local_centroids = [cent for site in args for cent in
                       args[site]]
    # and select k random clusters from the s*k pool
    np.random.shuffle(local_centroids)
    remote_centroids = local_centroids[:k]
    computation_output = dict(
        output=dict(
            work_dir=work_dir,
            config_file=config_file,
            centroids=remote_centroids,
            computation_phase="remote_init_centroids"
            ),
        success=True
    )
    return json.dumps(computation_output)


def remote_aggregate_optimizer(args):
    """
        Aggregate either with sum or mean
    """
    logging.info('REMOTE: Aggregate optimizer')
    computation_output = dict(
        output=dict(
            computation_phase="remote_aggregate_otpimizer"
            ),
        success=True
    )
    return json.dumps(computation_output)


def remote_optimization_step(args):
    """
        Aggregate with optimization step.
    """
    logging.info('REMOTE: Optimization step')
    computation_output = dict(
        output=dict(
            computation_phase="remote_optimization_step"
            ),
        success=True
    )
    return json.dumps(computation_output)


def remote_check_convergence(args):
    """
        Check convergence
    """
    logging.info('REMOTE: Check convergence')
    computation_output = dict(
        output=dict(
            computation_phase="remote_converged_true"
            ),
        success=True
    )
    computation_output = dict(
        output=dict(
            computation_phase="remote_converged_false"
            ),
        success=True
    )
    return json.dumps(computation_output)


def remote_aggregate_output(args):
    """
        Aggregate output.
    """
    logging.info('REMOTE: Aggregating input')
    computation_output = dict(
        output=dict(
            computation_phase="remote_aggregate_output"
            ),
        success=True
    )
    return json.dumps(computation_output)


if __name__ == '__main__':

    parsed_args = json.loads(sys.stdin.read())
    phase_key = list(list_recursive(parsed_args, 'computation_phase'))

    if not phase_key:
        computation_output = remote_init_env(**parsed_args['input'])
        sys.stdout.write(computation_output)
    elif 'local_init_centroids' in phase_key:
        computation_output = remote_init_centroids(parsed_args['input'])
        sys.stdout.write(computation_output)
    elif 'local_compute_optimizer' in phase_key:
        computation_output = remote_aggregate_optimizer(parsed_args['input'])
        sys.stdout.write(computation_output)
    elif 'remote_aggregate_otpimizer' in phase_key:
        computation_output = remote_optimization_step(parsed_args['input'])
        sys.stdout.write(computation_output)
    elif 'local_compute_clustering_2' in phase_key:
        computation_output = remote_check_convergence(parsed_args['input'])
        sys.stdout.write(computation_output)
    elif 'remote_converged_true' in phase_key:
        computation_outut = remote_aggregate_output(parsed_args['input'])
        sys.stdout.write(computation_output)
    else:
        raise ValueError('Oops')

