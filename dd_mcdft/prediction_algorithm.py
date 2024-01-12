import numpy as np


def predict_energy_closest(trainer, new_cluster,neg_cutoff):
    new_cluster = new_cluster.reshape(1, -1)
    new_dX_p = trainer.training_clusters - new_cluster
    new_dX_n = new_cluster - trainer.training_clusters
    neg_score = trainer.lof_model.score_samples(new_dX_n)
    neg_min_idx = trainer.closest_to_neg_one(neg_score)

    pos_score = trainer.lof_model.score_samples(new_dX_p)
    min_idx = trainer.closest_to_neg_one(neg_score)

    if (min_idx is None) or (np.abs(neg_score[min_idx]) > np.abs(neg_cutoff)):
        return None, neg_score[min_idx]

    else:
        pred_E_p = trainer.training_energy - trainer.ml_model.predict(new_dX_p)
        pred_E_n = trainer.training_energy + trainer.ml_model.predict(new_dX_n)
        pred_E = (pred_E_p[min_idx] + pred_E_n[min_idx]) / 2.0
        return pred_E, neg_score[min_idx]


def predict_energy_min(trainer, new_cluster,neg_cutoff):
    new_cluster = new_cluster.reshape(1, -1)
    new_dX_p = trainer.training_clusters - new_cluster
    new_dX_n = new_cluster - trainer.training_clusters
    neg_score = trainer.lof_model.score_samples(new_dX_n)
    pos_score = trainer.lof_model.score_samples(new_dX_p)

    neg_min_idx = np.argmin(neg_score)
    min_neg_score = neg_score[neg_min_idx]

    pos_min_idx = np.argmin(pos_score)
    min_pos_score = pos_score[pos_min_idx]
    avg_score = (min_neg_score + min_pos_score) / 2.0

    if min_neg_score < neg_cutoff:
        return None, avg_score

    else:
        pred_E_p = trainer.training_energy - trainer.ml_model.predict(new_dX_p)
        pred_E_n = trainer.training_energy + trainer.ml_model.predict(new_dX_n)
        pred_E = (pred_E_p[pos_min_idx] + pred_E_n[neg_min_idx]) / 2.0
        return pred_E, avg_score


def predict_energy_std(trainer, new_cluster,neg_cutoff):
    new_cluster = new_cluster.reshape(1, -1)
    new_dX_p = trainer.training_clusters - new_cluster
    new_dX_n = new_cluster - trainer.training_clusters
    neg_score = trainer.lof_model.score_samples(new_dX_n)
    # std_neg_score = np.std(neg_score)
    min_neg_score = np.min(neg_score)
    max_neg_score = np.max(neg_score)
    diff = np.abs(max_neg_score - min_neg_score)

    print(f"min: {min_neg_score:.3f} max:{max_neg_score:.3f} diff:{diff:.3f}")
    if diff > 0.6 or min_neg_score < neg_cutoff:
        return None, diff

    else:
        min_idx = np.argmin(neg_score)
        pred_E_p = trainer.training_energy - trainer.ml_model.predict(new_dX_p)
        pred_E_n = trainer.training_energy + trainer.ml_model.predict(new_dX_n)
        pred_E = (pred_E_p[min_idx] + pred_E_n[min_idx]) / 2.0
        return pred_E, diff


def predict_energy_closest_average(trainer, new_cluster,neg_cutoff):
    new_cluster = new_cluster.reshape(1, -1)
    new_dX_p = trainer.training_clusters - new_cluster
    new_dX_n = new_cluster - trainer.training_clusters
    neg_score = trainer.lof_model.score_samples(new_dX_n)
    min_neg_score = np.min(neg_score)

    if min_neg_score < neg_cutoff:
        return None, min_neg_score
    else:
        pred_E_p_all = trainer.training_energy - trainer.ml_model.predict(new_dX_p)
        pred_E_n_all = trainer.training_energy + trainer.ml_model.predict(new_dX_n)

        # Step 1: Find indices of neg_score that are less than zero
        indices_less_than_zero = np.where(neg_score < 0)[0]

        # Step 2: Make a new array of weights using 1/np.abs(neg_score+1) formula
        weights = 1 / np.abs(neg_score[indices_less_than_zero] + 1)

        # Step 3: Filter the pred_E_p_all and pred_E_n_all array by the indices found in step 1
        pred_E_p_all_filtered = pred_E_p_all[indices_less_than_zero]
        pred_E_n_all_filtered = pred_E_n_all[indices_less_than_zero]

        # Step 4: Calculate weighted average in pred_E_p_all_filtered and pred_E_n_all_filtered by the weight calculated in step 2
        weighted_avg_p = np.average(pred_E_p_all_filtered, weights=weights)
        weighted_avg_n = np.average(pred_E_n_all_filtered, weights=weights)

        # Step 5: Make average of these two weighted averages
        final_avg = (weighted_avg_p + weighted_avg_n) / 2

        return final_avg, min_neg_score


def predict_energy(trainer, new_cluster,neg_cutoff, algo):
    if algo == "min":
        return predict_energy_min(trainer, new_cluster, neg_cutoff)
    elif algo == "closest":
        return predict_energy_closest(trainer, new_cluster,neg_cutoff)
    elif algo == "closest_average":
        return predict_energy_closest_average(trainer, new_cluster,neg_cutoff)
    elif algo == "std":
        return predict_energy_std(trainer, new_cluster,neg_cutoff)
    else:
        raise Exception("Unknown algo keyword")
