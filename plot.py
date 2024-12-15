from abc import ABC, abstractmethod
from glob import glob
import json
import re
from pathlib import Path

from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import polars as pl


def reader_c2si(filepath, numeric_phones):
    with open(filepath) as f:
        data = json.load(f)

    def translator(index):
        if index == 0:
            return "silence"

        for phone, i in numeric_phones.items():
            if i == index:
                return phone

    labels = np.array([translator(value[0]) for value in data["labels"]])
    predicted = np.array([translator(value[0]) for value in data["predicted"]])

    return labels, predicted


class Confusion:
    def __init__(self, filepath, phones_subset=None, output_suffix=""):
        with open("confusion-matrix/numeric_phones.json") as f:
            self.numeric_phones = json.load(f)

        with open("confusion-matrix/phones.json") as f:
            self.phones = json.load(f)

        self.filepath = filepath
        self.output_png = self.filepath.split(".")[0] + output_suffix + ".pdf"
        self.label_names_organized = ["sil", "a", "Ê", "Û", "Ô", "u", "y", "i", "ã", "ɔ̃", "µ", "w", "ɥ", "j", "l", "ʁ",
                                      "n", "m", "ɲ", "p", "t", "k", "b", "d", "g", "f", "s", "ʃ", "v", "z", "ʒ"]
        self.phones_subset = phones_subset

        self.labels, self.predicted = reader_c2si(self.filepath, self.numeric_phones)

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

        return balanced_accuracy_score(
            [self.phones[a] for a in true_labels_no_sil], [self.phones[a] for a in predicted_labels_no_sil]
        )


class Plot(ABC):

    @abstractmethod
    def compute_graphs(self):
        pass

    def export(self, path: str):
        plt.savefig(f"{path}.pdf", bbox_inches='tight')
        plt.savefig(f"{path}.png", bbox_inches='tight', dpi=300)
        plt.close()


def get_json_path(model_name, short_id):
    possibilities = [short_id, short_id + "-1"]

    for p in possibilities:
        try:
            # print(f"results/{model_name}/2001/output_{p}.json")
            return Confusion(f"results/{model_name}/2001/output_{p}.json").get_balanced_accuracy()
        except FileNotFoundError:
            try:
                # print(f"results/{model_name}/2001/output-{p}-test.json")
                return Confusion(f"results/{model_name}/2001/output-{p}-test.json").get_balanced_accuracy()
            except FileNotFoundError:
                continue

    print(f"Unable to read patient {short_id}")


class AccuracyPerPatient(Plot):

    def __init__(self):
        models = {"Initial (Interspeech 2024)" :"unfrozen-cp-3k-large-accents", "Scenario D": "best-relu-long-context"}

        evaluations = pl.read_csv("../common/c2si-evaluations.csv").filter(pl.col('ID').str.contains("TIO"))

        # Create a column with SevDes and SevLec when SevDes is not specified, and sort by that column
        evaluations = evaluations.with_columns(
            pl.col("SevDes").fill_null(pl.col('SevLec')).alias('Sorting')
        ).sort(pl.col("Sorting"))

        results = {}
        for model in models.values():
            results[model] = []
            for patient_id in evaluations.select("ID").to_series().to_numpy():
                # print(patient_id)
                short_id = re.findall(r"\d+", patient_id)
                short_id = f"{int(short_id[0])}-{int(short_id[1])}" if int(short_id[1]) != 1 else f"{int(short_id[0])}"
                results[model].append(get_json_path(model, short_id))

            evaluations.insert_column(7, pl.Series(model, results[model]) * 100)

        self.evaluations = evaluations.with_columns(
            (pl.col("best-relu-long-context") - pl.col("unfrozen-cp-3k-large-accents")).alias("Difference")
        )

        self.evaluations = self.evaluations.filter(pl.col("Difference").is_not_null())
        print(self.evaluations)

    def compute_graphs(self):
        fig, ax = plt.subplots(figsize=(15, 6))

        ax.set_axisbelow(True)
        ax.grid(True, color='#cccccc', linestyle='--')

        ax.bar(self.evaluations.get_column("ID"), self.evaluations.get_column("Difference"))

        plt.xticks(fontsize=6, rotation=90)

        ax.set_ylabel('Balanced accuracy relative difference')
        ax.set_title('Relative difference in balanced accuracy for each patient.')

        arrow_style = {
            "head_width": 0.5,
            "head_length": 0.7,
            "color": "k"
        }
        plt.arrow(x=0, y=24, dx=112, dy=0, **arrow_style)
        plt.annotate('DES-Severity',
                     xy=(60, 24.2),
                     xytext=(-60, 2),
                     textcoords='offset points')

        plt.savefig("plots/accuracy_difference.pdf", bbox_inches='tight')


class HiddenLayerClassification(Plot):

    def __init__(self):
        self.accuracies = {}
        for path in glob("confusion-matrix/bref/array-*/output_test_accuracies.json"):
            with open(path) as f:
                hidden_layer = re.findall(r"\d+", path)[0]
                self.accuracies[hidden_layer] = json.load(f)["balanced_accuracy"]

        self.accuracies = self.get_data("confusion-matrix/bref/array-*/output_test_accuracies.json")
        self.c2si_dap = self.get_data("confusion-matrix/c2si/array-*/output_hc_dap_accuracies.json")
        self.c2si_lec = self.get_data("confusion-matrix/c2si/array-*/output_hc_lec_accuracies.json")

        self.data = {
            "BREF-Int": self.accuracies,
            "DAP (Healthy Controls)": self.c2si_dap,
            "LEC (Healthy Controls)": self.c2si_lec
        }

    def get_data(self, general_path: str):
        output = {}
        for path in glob(general_path):
            with open(path) as f:
                hidden_layer = int(re.findall(r"array-(\d+)", path)[0])
                output[hidden_layer] = json.load(f)["balanced_accuracy"]
                # print(path, hidden_layer, output[hidden_layer])

        return output

    def compute_graphs(self):
        fig, plots = plt.subplots(1, 1, figsize=(12, 5))

        x = [str(i) for i in range(25)]

        for key, data in self.data.items():
            values = [data[i] if i in data else 0.0 for i in range(25)]
            print(values)
            plots.plot(x, values, label=key)

        plots.legend(loc="upper right", reverse=True)
        plots.set_title('Balanced accuracies from hidden W2V2 layers')
        plots.set_xlabel('Hidden layer identifier')
        plots.set_ylabel('Balanced accuracy (in %)')
        plots.set_ylim([0, 100])
        plots.set_xlim([0, len(x) - 1])

        plots.set_axisbelow(True)
        plots.grid(True, color='#cccccc', linestyle='--')

        Path("plots/array/").mkdir(exist_ok=True, parents=True)
        self.export("plots/array/accuracies")


if __name__ == '__main__':
    # AccuracyPerPatient().compute_graphs()
    HiddenLayerClassification().compute_graphs()
