from abc import ABC, abstractmethod
from glob import glob
import json
import re
from pathlib import Path

from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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

    def     __init__(self):

        self.data = {
            "frozen": {
                "BREF-Int": self.get_data("confusion-matrix/bref/array-*/output_test_accuracies.json"),
                "DAP (Healthy Controls)": self.get_data("confusion-matrix/c2si/array-*/output_hc_dap_accuracies.json"),
                "LEC (Healthy Controls)": self.get_data("confusion-matrix/c2si/array-*/output_hc_lec_accuracies.json")
            },
            "fine-tuned": {
                "BREF-Int": self.get_data("confusion-matrix/bref/ft-array-*/output_test_accuracies.json"),
                "DAP (Healthy Controls)": self.get_data("confusion-matrix/c2si/ft-array-*/output_hc_dap_accuracies.json"),
                "LEC (Healthy Controls)": self.get_data("confusion-matrix/c2si/ft-array-*/output_hc_lec_accuracies.json")
            }
        }

        self.style = {
            "frozen": {
                "BREF-Int": ["tab:blue", "--"],
                "DAP (Healthy Controls)": ["tab:orange", "--"],
                "LEC (Healthy Controls)": ["tab:green", "--"]
            },
            "fine-tuned": {
                "BREF-Int": ["tab:blue", "-"],
                "DAP (Healthy Controls)": ["tab:orange", "-"],
                "LEC (Healthy Controls)": ["tab:green", "-"]
            }
        }

    def get_data(self, general_path: str):
        output = {}
        for path in glob(general_path):
            with open(path) as f:
                hidden_layer = int(re.findall(r"array-(\d+)", path)[0])
                output[hidden_layer] = json.load(f)["balanced_accuracy"]
                # print(path, output, output[hidden_layer])

        return output

    def compute_graphs(self):
        fig, plots = plt.subplots(1, 1, figsize=(12, 5))

        x = [str(i) for i in range(25)]

        print(self.data)
        for model_type, data in self.data.items():
            for key, d in data.items():
                values = [d[i] if i in d else 0.0 for i in range(25)]
                print(values)
                color, line_style = self.style[model_type][key]
                plots.plot(x, values, line_style, color=color, label=key if model_type=="fine-tuned" else "")

        plots.legend(loc="upper right", reverse=True)

        old_handles, labels = plots.get_legend_handles_labels()

        legend_elements = [Line2D([0], [0], ls="--", color='tab:blue', lw=2, label='W2V2 Frozen'),
                           Line2D([0], [0], ls='-', color='tab:blue', label='W2V2 fine-tuned')]

        plt.legend(handles=old_handles + legend_elements)

        plots.set_title('Balanced accuracies from hidden W2V2 layers')
        plots.set_xlabel('Hidden layer identifier')
        plots.set_ylabel('Balanced accuracy (in %)')
        plots.set_ylim([0, 100])
        plots.set_xlim([0, len(x) - 1])

        plots.set_yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

        plots.set_axisbelow(True)
        plots.grid(True, color='#cccccc', linestyle='--')

        Path("plots/array/").mkdir(exist_ok=True, parents=True)
        self.export("plots/array/accuracies")


class ContextLinePlot(Plot):

    def __init__(self):
        dimensions_zeros = ["", "-5-context", "-3-context", "-1-context"]
        dimensions_shuffle = ["", "-5-shuffle-context", "-3-shuffle-context", "-1-shuffle-context"]

        datasets = {
            "BREF-Int": "bref/best-relu-middle-segment-dropout{dim}/output_test",
            "LEC (HC)": "c2si/best-relu-middle-segment-dropout{dim}/output_hc_lec",
            "DAP (HC)": "c2si/best-relu-middle-segment-dropout{dim}/output_hc_dap",
            "LEC (Patients)": "c2si/best-relu-middle-segment-dropout{dim}/output_patients_lec",
            "DAP (Patients)": "c2si/best-relu-middle-segment-dropout{dim}/output_patients_dap"
        }

        self.error_margins = {
            "BREF-Int": 0.2,
            "LEC (HC)": 0.6,
            "DAP (HC)": 0.4,
            "LEC (Patients)": 0.2,
            "DAP (Patients)": 0.2
        }

        self.data_to_plot = {
            "When setting rest of context to 0": {
                k: [ContextLinePlot.read_data("confusion-matrix/" + path.format(dim=d) + "_accuracies.json") for d in
                    dimensions_zeros] for k, path in datasets.items()
            },
            "When shuffling contexts per batch": {
                k: [ContextLinePlot.read_data("confusion-matrix/" + path.format(dim=d) + "_accuracies.json") for d in
                    dimensions_shuffle] for k, path in datasets.items()
            }
        }

    @staticmethod
    def read_data(path):
        with open(path) as f:
            data = json.load(f)

        return data["balanced_accuracy"]

    def compute_graphs(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Influence of audio context around phone occurrence')

        for ax, (title, plotting_data) in zip((ax1, ax2), self.data_to_plot.items()):
            for key, d in plotting_data.items():
                ax.plot([0, 1, 2, 3], d, 'o-', label=key)
                ax.fill_between([0, 1, 2, 3], [a - self.error_margins[key] for a in d], [a + self.error_margins[key] for a in d], alpha=0.2)

            ax.set_title(title)
            if ax == ax2:
                ax.legend()

            ax.set_xticks([0, 1, 2, 3])
            ax.set_xticklabels(['Full context', '5/7 context', '3/7 context', '1/7 context'], fontsize=8)
            ax.set_xlabel("Portion of useful context")

            ax.set_yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
            ax.set_ylabel("Balanced accuracy (in %)")

            ax.set_ylim([0, 100])
            ax.set_xlim([0, 3])

            ax.set_axisbelow(True)
            ax.grid(True, color='#cccccc', linestyle='--')

        self.export("plots/context-experiments")


if __name__ == '__main__':
    # AccuracyPerPatient().compute_graphs()
    HiddenLayerClassification().compute_graphs()
    # ContextLinePlot().compute_graphs()