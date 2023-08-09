"""
@CreateTime: 2023-06-30 13:51:47
@LastEditTime: 2023-06-30 13:51:57
@Description: 
"""
import os
import os.path as osp
import argparse
import json
import random
from datetime import datetime

import numpy as np
import torch

from runner import BaseRunner, PPRRunner, SSLRunner
import utils

root_path = utils.root_path


def set_seed(seed=2023):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=osp.join(root_path),
        help="JSON file for configuration",
    )
    parser.add_argument("-g", "--gpu", type=int, default=2, help="Using gpu #")
    parser.add_argument("-d", "--data_name", type=str, default="gowalla", help="dataset name")
    parser.add_argument("-s", "--seed", type=int, default=352, help="random seed")
    parser.add_argument("-m", "--method", type=str, default="ssl", help="method name")
    args = parser.parse_args()

    # load args
    with open(args.config) as rfile:
        s = rfile.read()
    config = json.loads(s)

    config["device"] = torch.device("cuda:{}".format(args.gpu))
    config["method"] = args.method
    config["data"]["name"] = args.data_name
    config["model"]["seed"] = args.seed

    return config


def get_config_parser():
    """
    @description: get config with parser
    """
    config = parser()
    set_seed(config["model"]["seed"])

    # update data-wise config
    path_config_data = osp.join(root_path, "config", config["data"]["name"] + ".json")
    with open(path_config_data) as rfile:
        s = rfile.read()
    config_data = json.loads(s)
    config["model"].update(config_data["model"])

    return config


def get_config_static(method, data_name, seed, g=0):
    """
    @description: get static config based on m, d, s
    """
    # load config
    if method in ["topppr", "tarppr", "toptarppr", "ssl"]:
        path_config = osp.join(root_path, "config", method + ".json")
    else:
        path_config = osp.join(root_path, "config", "lg.json")
    with open(path_config) as rfile:
        s = rfile.read()
    config = json.loads(s)
    config["method"] = method
    config["device"] = torch.device("cuda:{}".format(g))
    config["data"]["name"] = data_name
    config["model"]["seed"] = seed

    # update data-wise config
    path_config_data = osp.join(root_path, "config", config["data"]["name"] + ".json")
    with open(path_config_data) as rfile:
        s = rfile.read()
    config_data = json.loads(s)
    config["model"].update(config_data["model"])

    return config


def get_runner(config):
    method = config["method"]
    if method == "ssl":
        runner = SSLRunner(config)

    return runner
