from dd_mcdft.cluster_expansion import cluster_expansion
from dd_mcdft.train_model import ML_trainer
from dd_mcdft.test_model import ML_tester
from ase.build import bulk
from ase.io import read
import numpy as np
from copy import copy
from dd_mcdft.extract_summery import extract_data_summery
import pandas as pd
import json

# primitive_structure = bulk("W", "bcc", a=3.0787)
# cutoffs = [10.00, 6.00]

# chemical_symbols = {
#     "HEA1": ["W", "Ti", "Cr", "Ta"],
#     "HEA2": ["W", "Ti", "V", "Ta"],
#     "SA1": ["W", "Ti", "Cr"],
#     "SA2": ["W", "Cr", "Y"],
# }


alloy = "HEA1"
algo = "min"
train_dx=True

csv_file = f"./database/{alloy}_100k_non_seq1.csv"

train_sizes = [
    25,
    50,
    75,
    100,
    125,
    150,
    175,
    200,
    225,
    250,
    275,
    300,
    350,
    400,
    450,
    500,
]

neg_cutoffs = [-3.5, -3.25, -3.0, -2.75, -2.5]

train_sizes=[500]
neg_cutoffs=[-2.5]

# cluster_calculator = cluster_expansion(
#     primitive_structure, cutoffs, chemical_symbols[alloy]
# )

all_data=[]
for train_size in train_sizes:
    trainer_master = ML_trainer(train_size, csv_file, train_dx)
    for neg_cutoff in neg_cutoffs:
        trainer=copy(trainer_master)
        out_filename = (
            f"./model_test_data_x/{alloy}_{train_size}_{neg_cutoff}_{algo}_x.dat"
        )
        trainer.train_model()
        tester = ML_tester(trainer, out_filename, algo, neg_cutoff,train_dx)
        tester.test()
        
        data=extract_data_summery(out_filename, train_size, neg_cutoff, csv_file)
        data["n_feature"]=trainer_master.all_clusters.shape[1]
        data["train_dx"]=train_dx
        data["algo"]=algo
        print(json.dumps(data, indent=4))
        all_data.append(data)

df = pd.DataFrame(all_data)
df.to_csv(f"./model_test_data_x/{alloy}_{algo}_summery_x.csv", index=False)