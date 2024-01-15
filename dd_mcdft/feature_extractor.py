from ase.io import read
from dd_mcdft.cluster_expansion import cluster_expansion
import numpy as np



def extract_energies_from_trajectory(cs, in_filename, out_filename):
    traj = read(f"{in_filename}", index=":")
    energies = list(map(lambda atoms: atoms.get_potential_energy(), traj))
    times = list(map(lambda atoms: atoms.info["time"], traj))
    clusters = list(map(lambda atoms: cs.extract_feature([atoms])[0], traj))

    clusters = np.array(clusters)
    energies = np.array(energies).reshape(-1, 1)
    times = np.array(times).reshape(-1, 1)
    print(f"Cluster data shape: {clusters.shape}")
    print(f"Energy Data shape: {energies.shape}")
    data = np.hstack((clusters, energies, times))
    print(f"Combine data shape : {data.shape}")

    np.savetxt(f"{out_filename}", data, delimiter=",")
