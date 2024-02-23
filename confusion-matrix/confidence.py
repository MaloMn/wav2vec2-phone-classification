from confidence_intervals import evaluate_with_conf_int
from sklearn.metrics import balanced_accuracy_score
import json
import numpy as np
from confusion import reader_bref, reader_c2si

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class Confidence:
    def __init__(self, filepath):
        with open("numeric_phones.json") as f:
            self.numeric_phones = json.load(f)

        with open("phones.json") as f:
            self.phones = json.load(f)

        self.filepath = filepath

        if ".txt" in self.filepath:
            self.labels, self.predicted = reader_bref(self.filepath, self.numeric_phones)
        elif ".json" in self.filepath:
            self.labels, self.predicted = reader_c2si(self.filepath, self.numeric_phones)
        else:
            raise Exception("This dataset is not supported yet.")

        self.balanced = self.get_balanced_accuracy()

    def get_balanced_accuracy(self):
        # Remove silences
        silence_indexes = np.where(self.labels == "silence")[0]
        true_labels_no_sil = np.delete(self.labels, silence_indexes)
        predicted_labels_no_sil = np.delete(self.predicted, silence_indexes)

        # Replacing the silence predicted by a random different phone
        for i, ph in enumerate(predicted_labels_no_sil):
            if ph == "silence":
                for sub_ph in ["aa", "bb"]:
                    if sub_ph != true_labels_no_sil[i]:
                        predicted_labels_no_sil[i] = sub_ph
                        break

        true = np.array([self.phones[a] for a in true_labels_no_sil])
        predicted = np.array([self.phones[a] for a in predicted_labels_no_sil])
        return evaluate_with_conf_int(predicted, balanced_accuracy_score, true, num_bootstraps=1000, alpha=5)


def print_confidence(confidence):
    value, (mini, maxi) = confidence
    print(f"\t→ {round(value*100, 2)} lies in [{round(mini*100, 2)}; {round(maxi*100, 2)}]")


def iterate(*folders):
    with open("confidences.json", "r") as f:
        conf = json.load(f)

    for folder in folders:
        print(folder)
        bref_int = Confidence(f"bref/{folder}/output_test.json").balanced
        print_confidence(bref_int)
        c2si_dap = Confidence(f"c2si/{folder}/output_hc_dap.json").balanced
        print_confidence(c2si_dap)
        c2si_lec = Confidence(f"c2si/{folder}/output_hc_lec.json").balanced
        print_confidence(c2si_lec)

        conf[folder] = {
            "bref-int": {"value": bref_int[0], "min": bref_int[1][0], "max": bref_int[1][1]},
            "c2si-dap": {"value": c2si_dap[0], "min": c2si_dap[1][0], "max": c2si_dap[1][1]},
            "c2si-lec": {"value": c2si_lec[0], "min": c2si_lec[1][0], "max": c2si_lec[1][1]}
        }

    with open("confidences.json", "w") as f:
        json.dump(conf, f, indent=4)


if __name__ == '__main__':
    iterate("unfrozen-cp-14k-large-accents", "unfrozen-cp-14k-light-accents", "unfrozen-cp-3k-large-accents",
            "unfrozen-cp-3k-base-accents")

    #  "frozen", "unfrozen", "unfrozen-14k-light", "unfrozen-3k-large", "unfrozen-3k-base",
