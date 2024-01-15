from ase.build import bulk
import time
import  numpy as np
from dd_mcdft.cluster_expansion import cluster_expansion
from dd_mcdft.feature_extractor import extract_energies_from_trajectory 

primitive_structure = bulk("W", "bcc", a=3.0787)
chemical_symbols = ["Ta", "W", "Ti", "Cr"]
alloy="HEA1_100k_non_seq1"
cutoffs_x = np.linspace(7.5, 10.5, num=4)


for x in cutoffs_x:
    cutoffs = [8.5,x]
    in_filename = f"../database/{alloy}.traj"
    out_filename = f"./N_feature_database/{alloy}_8.5_{x}.csv"
    cs = cluster_expansion(primitive_structure, cutoffs, chemical_symbols)
    t1 = time.time()
    extract_energies_from_trajectory(cs, in_filename, out_filename)
    t2 = time.time()
    print(f"Time taken to extarct feature: {t2-t1} sec")