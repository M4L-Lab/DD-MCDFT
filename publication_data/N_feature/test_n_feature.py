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



alloy = "HEA1_100k_non_seq1"
algo = "min"
train_dx=False


cutoffs_x = np.linspace(0.5, 9.5, num=10)

train_size = 500
neg_cutoffs = [-3.5, -3.25, -3.0, -2.75, -2.5]


all_data=[]
for x in cutoffs_x:
    csv_file=f"./N_feature_database/{alloy}_8.5_{x}.csv"
    trainer_master = ML_trainer(train_size, csv_file, train_dx)
    for neg_cutoff in neg_cutoffs:
        trainer=copy(trainer_master)
        out_filename = (
            f"./N_feature_500/{alloy}_{train_size}_{neg_cutoff}_{algo}_8.5_{x}.dat"
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
df.to_csv(f"./N_feature_500/{alloy}_{algo}_summery_8.5_x.csv", index=False)