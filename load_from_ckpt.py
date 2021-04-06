import os
import torch

from saifooler.attacks.pgd_attack import PGDAttack


logs_path = "./logs"

for root, dirs, files in os.walk(logs_path):
    for name in files:
        if ".ckpt" in name:
            ckpt_path = os.path.join(root, name)

            x = torch.load(ckpt_path)
            #att = PGDAttack.load_from_checkpoint(ckpt_path)
            print(x)

