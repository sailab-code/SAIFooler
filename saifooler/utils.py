import argparse

import torch

from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams

def greyscale_heatmap(images):
    """

    :param images: shape NxWxH
    :return:
    """
    # rescale pixels in 0..1
    return images / images.max(1, keepdim=True)[0].max(2, keepdim=True)[0]




class SummaryWriter(SummaryWriter):
    def add_hparams(self, hparam_dict, metric_dict, hparam_domain_discrete=None, run_name=None):
        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError('hparam_dict and metric_dict should be dictionary.')
        exp, ssi, sei = hparams(hparam_dict, metric_dict, hparam_domain_discrete)

        logdir = self._get_file_writer().get_logdir()

        # import os
        # if not run_name:
        #     import time
        #
        #     run_name = str(time.time())
        # logdir = os.path.join(self._get_file_writer().get_logdir(), run_name)
        with SummaryWriter(log_dir=logdir) as w_hp:
            w_hp.file_writer.add_summary(exp)
            w_hp.file_writer.add_summary(ssi)
            w_hp.file_writer.add_summary(sei)
            for k, v in metric_dict.items():
                w_hp.add_scalar(k, v)


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')