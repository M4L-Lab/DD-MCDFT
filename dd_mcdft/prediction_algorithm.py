import numpy as np


def predict_energy_closest(trainer, new_cluster,neg_cutoff):
    new_cluster = new_cluster.reshape(1, -1)
    new_dX_p = trainer.training_clusters - new_cluster
    #new_dX_n = new_cluster - trainer.training_clusters
    neg_score = trainer.lof_model.score_samples(new_dX_p)
    min_idx = trainer.closest_to_neg_one(neg_score)

    if (min_idx is None) or (np.abs(neg_score[min_idx]) > np.abs(neg_cutoff)):
        return None, neg_score[min_idx]

    else:
        pred_E_p = trainer.training_energy - trainer.ml_model.predict(new_dX_p)
        #pred_E_n = trainer.training_energy + trainer.ml_model.predict(new_dX_n)
        #pred_E = (pred_E_p[min_idx] + pred_E_n[min_idx]) / 2.0
        return pred_E_p[min_idx], neg_score[min_idx]

def predict_energy_closest_x(trainer, new_cluster,neg_cutoff):
    new_cluster = new_cluster.reshape(1, -1)
    neg_score = trainer.lof_model.score_samples(new_cluster)
    min_idx = trainer.closest_to_neg_one(neg_score)

    new_dX_p = trainer.training_clusters[min_idx] - new_cluster
    new_dX_n = new_cluster - trainer.training_clusters[min_idx]
    

    #if np.min(neg_score) < neg_cutoff:
    if (min_idx is None) or (np.abs(neg_score[min_idx]) > np.abs(neg_cutoff)):
        return None, neg_score[min_idx]

    else:
        pred_E_p = trainer.training_energy[min_idx] - trainer.ml_model.predict(new_dX_p)
        pred_E_n = trainer.training_energy[min_idx] + trainer.ml_model.predict(new_dX_n)
        pred_E = (pred_E_p + pred_E_n) / 2.0
        return pred_E[0], neg_score[min_idx]


def predict_energy_min(trainer, new_cluster,neg_cutoff):
    new_cluster = new_cluster.reshape(1, -1)
    new_dX_p = trainer.training_clusters - new_cluster
    # new_dX_n = new_cluster - trainer.training_clusters
    # neg_score = trainer.lof_model.score_samples(new_dX_n)
    pos_score = trainer.lof_model.score_samples(new_dX_p)

    # neg_min_idx = np.argmin(neg_score)
    # min_neg_score = neg_score[neg_min_idx]

    pos_min_idx = np.argmin(pos_score)
    min_pos_score = pos_score[pos_min_idx]
    #avg_score = (min_neg_score + min_pos_score) / 2.0

    if min_pos_score < neg_cutoff:
        return None, min_pos_score

    else:
        pred_E_p = trainer.training_energy - trainer.ml_model.predict(new_dX_p)
        #pred_E_n = trainer.training_energy + trainer.ml_model.predict(new_dX_n)
        #pred_E = (pred_E_p[pos_min_idx] + pred_E_n[neg_min_idx]) / 2.0
        return pred_E_p[pos_min_idx], min_pos_score

def predict_energy_min_x(trainer, new_cluster,neg_cutoff):
    new_cluster = new_cluster.reshape(1, -1)
    neg_score = trainer.lof_model.score_samples(new_cluster)

    neg_min_idx = np.argmin(neg_score)
    min_neg_score = neg_score[neg_min_idx]

    new_dX_p = trainer.training_clusters[neg_min_idx] - new_cluster
    #new_dX_n = new_cluster - trainer.training_clusters[neg_min_idx]

    if min_neg_score < neg_cutoff:
        return None, min_neg_score 

    else:
        pred_E_p = trainer.training_energy[neg_min_idx] - trainer.ml_model.predict(new_dX_p)
        #pred_E_n = trainer.training_energy[neg_min_idx] + trainer.ml_model.predict(new_dX_n)
        #pred_E = (pred_E_p + pred_E_n) / 2.0
        return pred_E_p[0], min_neg_score

def predict_energy(trainer, new_cluster,neg_cutoff, algo, train_dx):
    if algo == "min":
        if train_dx:
            return predict_energy_min(trainer, new_cluster, neg_cutoff)
        else:
            return predict_energy_min_x(trainer, new_cluster, neg_cutoff)

    elif algo == "closest":
        if train_dx:
            return predict_energy_closest(trainer, new_cluster,neg_cutoff)
        else:
            return predict_energy_closest_x(trainer, new_cluster,neg_cutoff)
    else:
        raise Exception("Unknown algo keyword")
        

