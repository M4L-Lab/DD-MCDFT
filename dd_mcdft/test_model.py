from dd_mcdft.file_io import FileIO
from dd_mcdft.prediction_algorithm import predict_energy
from sklearn.metrics import r2_score
import numpy as np
import time


class ML_tester:
    def __init__(self, trainer, out_filename, algo, neg_cutoff):
        self.trainer = trainer
        self.neg_cutoff=neg_cutoff
        self.out_file = FileIO(out_filename)
        self.algo = algo

    def test(self):
        for i in range(self.trainer.train_size, len(self.trainer.all_clusters)):
            t1 = time.time()
            new_cluster = self.trainer.all_clusters[i]
            pred_E, neg_score = predict_energy(
                self.trainer, new_cluster,self.neg_cutoff, algo=self.algo
            )
            real_E = self.trainer.all_energy[i]

            if pred_E is None:
                self.trainer.refit(new_cluster, real_E)
                delta_time = self.trainer.all_time[i]
                msg = f"{i:5} {neg_score:10.5f} {real_E:10.5f} {str(np.nan):>11} {str(np.nan):>11} 1 {delta_time:20.3f}"
            else:
                delta_time = time.time() - t1
                err = real_E - pred_E
                msg = f"{i:5} {neg_score:10.5f} {real_E:10.5f} {pred_E:10.5f} {err:10.5f} 0 {delta_time:20.3f}"
            
            self.out_file.write_formatted_message(msg)
            if(i%10==0):
                print('-', end='', flush=True)

