import numpy as np
from ase.io import read
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd


def get_dft_times(training_csv_filename):
    data = np.genfromtxt(
        training_csv_filename,
        delimiter=",",
        missing_values="nan",
        filling_values=np.nan,
    )
    dft_times = data[:, -1] / 3600

    return dft_times


def get_times(data, train_size, training_csv_filename):
    DD_times = data[:, -1] / 3600
    dft_times = get_dft_times(training_csv_filename)
    hybrid_times = np.append(dft_times[:train_size], DD_times)
    cumsum_hybrid_times = np.nancumsum(hybrid_times)
    time_required = cumsum_hybrid_times[-1]
    return time_required


# msg = f"{i:5} {neg_score:10.5f} {real_E:10.5f} {pred_E:10.5f} {err:10.5f} 0 {delta_time:20.3f}"


def extract_data_summery(filename, train_size, neg_cutoff, training_csv_filename):
    prop_dict = {}
    data = np.genfromtxt(f"{filename}", missing_values="nan", filling_values=np.nan)
    rows_without_nan = ~np.isnan(data[:, 3])
    data_clean = data[rows_without_nan]

    extra_dft = np.sum(np.isnan(data[:, 3]))
    r2 = r2_score(data_clean[:, 2], data_clean[:, 3])
    rmse = mean_squared_error(data_clean[:, 2], data_clean[:, 3]) * 1000 / 128
    
    bias_factor = np.sum(data_clean[:, 2] - data_clean[:, 3])
    time_saved = get_times(data, train_size, training_csv_filename)

    max_error = np.max(data_clean[:, 2] - data_clean[:, 3])
    prop_dict["train_size"] = int(train_size)
    prop_dict["neg_score_cutoff"] = neg_cutoff
    prop_dict["extra_dft"] = int(extra_dft)
    prop_dict["r2"] = round(r2,4)
    prop_dict["rmse"] = round(rmse,4)
    prop_dict["bias_factor"] = round(bias_factor,4)
    prop_dict["max_error"] = round(max_error,4)
    prop_dict["time_saved"] = round(time_saved,4)
    return prop_dict