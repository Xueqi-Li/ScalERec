"""
@CreateTime: 2023-06-30 15:14:28
@LastEditTime: 2023-07-01 13:06:37
@Description: 
"""
import os
import torch
import runner_utils


def train():
    # config = runner_utils.get_config_static("ssl", "gowalla", 352, g=0)
    config = runner_utils.get_config_parser()
    runner = runner_utils.get_runner(config)

    print(config)
    runner.train()


def get_epoch_time(list_data=None, list_method=None):
    if list_data is None:
        list_data = ["gowalla", "yelp2018", "book", "taobao"][2:]
    if list_method is None:
        list_method = [
            # "lg",
            # "sgl",
            # "simgcl1",
            # "appnp",
            # "topppr",
            # "sage3",
            # "sage10",
            # "cluster100",
            # "cluster1000",
            # "saint1000",
            # "saint10000",
            "ssl",
            # "toptarppr",
        ]
    dict_method_name = {
        "lg": "LightGCN",
        "sgl": "SGL",
        "simgcl1": "SimGCL",
        "simgcl2": "SimGCL2",
        "topppr": "PPRGo",
        "appnp": "APPNP",
        "sage3": "GraphSAGE-3",
        "sage10": "GraphSAGE-10",
        "cluster100": "Cluster-GCN-100",
        "cluster1000": "Cluster-GCN-1000",
        "saint1000": "GraphSAINT-1000",
        "saint10000": "GraphSAINT-10000",
        "ssl": "PPRSR",
        "toptarppr": "PPRSR-CL",
    }

    for m in list_method:
        list_t = [dict_method_name[m]]
        for d in list_data:
            print(m, d)
            config = runner_utils.get_config_static(m, d, 352, g=0)
            runner = runner_utils.get_runner(config)
            try:
                if d in ["book", "taobao"]:
                    skip, active = 1, 3
                    t = runner.get_epoch_time(skip, active)
                    list_t.append("{:.4f}".format(t / 60))
                else:
                    skip, active = 1, 10
                    t = runner.get_epoch_time(skip, active)
                    list_t.append("{:.4f}".format(t))
            except Exception as e:
                print(e)
                list_t.append("-")
        print(" & ".join(list_t) + "\\\\")


if __name__ == "__main__":
    train()
    # # get_epoch_time(list_data=["gowalla", "yelp2018"])
    # print("large dataset")
    # get_epoch_time(list_data=["book", "taobao"])
