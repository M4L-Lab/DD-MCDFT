from dd_mcdft.cluster_expansion import cluster_expansion
from dd_mcdft.train_model import ML_trainer
from dd_mcdft.test_model import ML_tester
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import Ridge, Lasso
from ase.build import bulk
from ase.io import read

primitive_structure = bulk("W", "bcc", a=3.0787)
cutoffs = [10.00, 6.00]

chemical_symbols = {
    "HEA1": ["W", "Ti", "Cr", "Ta"],
    "HEA2": ["W", "Ti", "V", "Ta"],
    "SA1": ["W", "Ti", "Cr"],
    "SA2": ["W", "Cr", "Y"],
}


alloy = "HEA1"
algo = "min"

csv_file = f"./{alloy}_100k_non_seq1.csv"

train_sizes = [25, 50, 100, 200, 300, 500]
neg_cutoffs = [-4.0, -3.75, -3.5, -3.25, -3.0, -2.75, -2.5, -2.25, -2.0]

cluster_calculator = cluster_expansion(
    primitive_structure, cutoffs, chemical_symbols[alloy]
)

lof_model = LocalOutlierFactor(n_neighbors=5, novelty=True)
ml_model = Ridge(alpha=0.001)


trainer = ML_trainer(25, csv_file, lof_model, ml_model, -2.5)
out_filename = "test.dat"
trainer.train_model()
tester = ML_tester(trainer, out_filename)
tester.test()
