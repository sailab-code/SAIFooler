import os
import torch

import shutil

from saifooler.attacks.pgd_attack import PGDAttack


original_logs_path = "/media/users/tiezzi/Projects/inProgress/SAIFooler/logs_09_march"
my_logs_path = "./logs_09_march"


def replace_path(path):
    return path.replace(original_logs_path, my_logs_path)


for root, dirs, files in os.walk(original_logs_path):
    for name in files:
        src_path = os.path.join(root, name)
        dst_path = replace_path(src_path)
        dst_dir = os.path.dirname(dst_path)

        if ".json" in name and not "mat_def" in name:
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copyfile(src_path, dst_path)

        if ".ckpt" in name:
            ckpt_path = os.path.join(root, name)
            ckpt = torch.load(ckpt_path)
            delta = ckpt['state_dict']['delta']
            os.makedirs(dst_dir, exist_ok=True)
            torch.save(delta, dst_path)
