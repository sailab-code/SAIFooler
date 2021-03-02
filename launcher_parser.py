import argparse
import seaborn as sns
import os
import pandas as pd
import json

from itertools import product

if __name__ == '__main__':

    # hyperparams

    EPS = [5, 20, 30, 50]
    ALPHA = [0.001, 0.01, 0.1, 1.0]
    CLASSIFIER = ["inception", "mobilenet"]
    SALIENCY = [True, False]
    SALIENCY_THRESH = [0.01, 0.02, 0.05, 0.5]

    # df_global = pd.Dataframe(columns=["eps", "alpha", "model", "saliency", "saliency_threshold",
    #                                   "pytorch_no_attack", "pytorch_attack", "sailenv_no_attack",
    #                                   "sailenv_attack"])

    acc_list = []

    for eps_, alpha_, classifier_, saliency_ in product(EPS, ALPHA, CLASSIFIER, SALIENCY,
                                                        ):

        exp_name_base = f"eps_{eps_}__alpha_{alpha_}__model_{classifier_}_saliency_{saliency_}"

        params_dict = {"eps": eps_,
                       "alpha": alpha_,
                       "model": classifier_,
                       "saliency": saliency_}

        if saliency_:
            saliency_thresh_ = SALIENCY_THRESH
        else:
            saliency_thresh_ = [-1]

        for s_th_ in saliency_thresh_:

            if saliency_:
                exp_name = exp_name_base + f"_saliency_thresh_{s_th_}"
                params_dict["saliency_threshold"] = s_th_
            else:
                exp_name = exp_name_base

            model_name = classifier_

        try:
            log_folder = os.path.join("logs_24_02", "pgd", exp_name, "default", "version_0", "summary.json")
            with open(log_folder) as json_file:
                acc = json.load(json_file)

            global_dict = {**params_dict, **acc}
            acc_list.append(global_dict)
        except:
            pass

    df_global = pd.DataFrame(acc_list)
    df_global.to_csv("acc_global.csv")
