import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import Ridge, Lasso


class ML_trainer:
    """Read csv file, make dataset from based on train_size
    and save the model"""

    def __init__(self, train_size, csv_file, train_dx=True):
        self.csv_file = csv_file
        self.train_size = train_size
        self.train_dx=train_dx
        self.lof_model = LocalOutlierFactor(n_neighbors=5, novelty=True)
        self.ml_model = Ridge(alpha=0.001)
        self.training_atoms_idx = []
        self._get_data_from_csv()
        self.get_training_data()

    def _get_data_from_csv(self):
        data = np.loadtxt(self.csv_file, delimiter=",")
        self.all_clusters = data[:, :-2]
        self.all_energy = data[:, -2]
        self.all_time = data[:, -1]
        self.training_clusters = data[: self.train_size, :-2]
        self.training_energy = data[: self.train_size, -2]


    def get_training_data(self):
        dX_train = []
        dY_train = []
        for i in range(self.train_size):
            for j in range(self.train_size):
                dX_train.append(self.training_clusters[i] - self.training_clusters[j])
                dY_train.append(self.training_energy[i] - self.training_energy[j])
        self.dX_train = np.array(dX_train)
        self.dY_train = np.array(dY_train)

    def train_model(self):

        if self.train_dx:
            self.lof_model.fit(self.dX_train)
        else:
            self.lof_model.fit(self.training_clusters)

        self.ml_model.fit(self.dX_train, self.dY_train)

    def closest_to_neg_one(self, arr):
        # filter negative elements
        neg_elements = arr[arr < 0]
        if len(neg_elements) == 0:
            return np.argmin(arr)
        # find the closest to -1
        closest_element = neg_elements[np.argmin(np.abs(neg_elements + 1))]
        # get the index of the closest element
        index = np.where(arr == closest_element)
        return index[0][0]

    def refit(self, new_cluster, new_energy, train_dx):
        new_cluster = new_cluster.reshape(1, -1)
        new_energy = np.array([new_energy])
        new_dX = self.training_clusters - new_cluster
        new_dY = self.training_energy - new_energy
        self.ml_model.fit(new_dX, new_dY)
        self.ml_model.fit(-1 * new_dX, -1 * new_dY)
        self.training_clusters = np.append(self.training_clusters, new_cluster, axis=0)
        self.training_energy = np.append(self.training_energy, new_energy, axis=0)
        if train_dx: 
            self.lof_model.fit(new_dX)
            self.lof_model.fit(-1 * new_dX)
        else:
            self.lof_model.fit(self.training_clusters)