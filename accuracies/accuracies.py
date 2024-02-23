import re

from sklearn.metrics import confusion_matrix, balanced_accuracy_score, accuracy_score
import matplotlib.pyplot as plt
import itertools
import numpy as np
import json


def warn(*args, **kwargs):
    pass

import warnings

warnings.warn = warn

def reader_bref(filepath, numeric_phones):
    def translator(character):
        character = character.replace(" ; ", "").replace(" ", "").replace("<eps>", "")
        index = int(character)

        if index == 0:
            return "silence"

        for phone, i in numeric_phones.items():
            if i == index:
                return phone

    with open(filepath) as f:
        lines = [line.replace("\n", "") for line in f.readlines()]

    true_labels = np.array([translator(value) for value in lines[13::5]])
    predicted_labels = np.array([translator(value) for value in lines[15::5]])

    return true_labels, predicted_labels


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


def linear_mapping_function(x1, y1, x2, y2):
    # Calculate the slope and intercept of the linear equation
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1

    # Define the linear mapping lambda function
    return lambda x: slope * x + intercept


class Confusion:
    def __init__(self, filepath, phones_subset=None, output_suffix=""):
        with open("numeric_phones.json") as f:
            self.numeric_phones = json.load(f)

        with open("phones.json") as f:
            self.phones = json.load(f)

        self.filepath = filepath
        self.output_png = self.filepath.split(".")[0] + output_suffix + ".png"
        self.label_names_organized = ["sil", "a", "Ê", "Û", "Ô", "u", "y", "i", "ã", "ɔ̃", "µ", "w", "ɥ", "j", "l", "ʁ",
                                      "n", "m", "ɲ", "p", "t", "k", "b", "d", "g", "f", "s", "ʃ", "v", "z", "ʒ"]
        self.phones_subset = phones_subset

        if ".txt" in self.filepath:
            self.labels, self.predicted = reader_bref(self.filepath, self.numeric_phones)
        elif ".json" in self.filepath:
            self.labels, self.predicted = reader_c2si(self.filepath, self.numeric_phones)
        else:
            raise Exception("This dataset is not supported yet.")

        self.accuracy, self.balanced = self.get_accuracies()
        # self.confusion_matrix = self.compute_confusion_matrix()
        # self.plot_confusion_matrix()

    def get_accuracies(self, save=True):
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

        accuracy = accuracy_score(
            [self.phones[a] for a in true_labels_no_sil], [self.phones[a] for a in predicted_labels_no_sil]
        )

        balanced_accuracy = balanced_accuracy_score(
            [self.phones[a] for a in true_labels_no_sil], [self.phones[a] for a in predicted_labels_no_sil]
        )

        if save:
            with open(self.output_png.split('.')[0] + "_accuracies.json", "w+") as f:
                json.dump({
                    "accuracy": round(accuracy*100, 2),
                    "balanced_accuracy": round(balanced_accuracy * 100, 2)
                }, f, indent=4)

        return accuracy, balanced_accuracy

    def compute_confusion_matrix(self):
        cm = confusion_matrix([self.phones[a] for a in self.labels], [self.phones[a] for a in self.predicted], normalize="true", labels=self.label_names_organized)

        for i in range(cm.shape[0]):
            if np.count_nonzero(cm[i, :]) == 0:
                cm[i, :] = np.nan

        if self.phones_subset is not None:
            keep_indexes = [self.label_names_organized.index(a) for a in self.phones_subset]
            cm = cm[:, keep_indexes][keep_indexes, :]

        return cm

    def plot_confusion_matrix(self, savefig=True, file='', cmap=plt.cm.Greys):
        fig, ax = plt.subplots(dpi=125)
        fig.set_size_inches(15, 12, forward=True)

        rect = plt.Rectangle((-0.5, -0.5), self.confusion_matrix.shape[1], self.confusion_matrix.shape[0], linewidth=2, edgecolor='black', facecolor='none')
        ax.add_patch(rect)

        im = ax.imshow(self.confusion_matrix, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
        ax.set_title(file, fontsize=20)
        # plt.colorbar(im, ax=ax)

        if "bref/" in self.filepath:
            title = "BREF"
        else:
            title = re.search(r'output_(.*)\.json', self.filepath).group(1).replace("_", " ")
            # title = self.filepath.replace("c2si/output_", "").replace(".json", "").replace("_", " ")
        # ax.set_title("Confusion Matrix - " + title, fontsize=20)

        labels = self.phones_subset if self.phones_subset is not None else self.label_names_organized

        tick_marks = np.arange(len(labels))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)

        fontsize = linear_mapping_function(1, 27, 31, 14)(len(labels))
        ax.set_xticklabels([f"/{a}/" for a in labels], fontsize=fontsize, weight='bold')
        ax.set_yticklabels([f"/{a}/" for a in labels], fontsize=fontsize, weight='bold')

        ax.grid(False)

        fontsize = linear_mapping_function(1, 23, 31, 10)(len(labels))
        for i, j in itertools.product(range(self.confusion_matrix.shape[0]), range(self.confusion_matrix.shape[1])):
            color = "white" if self.confusion_matrix[i, j] > 0.6 else "black"
            ax.text(j, i, f'{self.confusion_matrix[i, j] * 100:.1f}',
                    ha="center", va="center", color=color, fontsize=fontsize, weight='bold')

        plt.tight_layout(pad=3)
        ax.set_ylabel('True labels', fontsize=18, weight='bold')
        ax.set_xlabel('Predicted labels', fontsize=18, weight='bold')

        if savefig:
            plt.savefig(self.output_png, bbox_inches='tight', pad_inches=0.1)
            print(f"Confusion matrix was saved at {self.output_png}")


if __name__ == '__main__':
    import glob, re, csv

    with open("c2si/patient-accuracies.csv", 'w', newline='') as csv_file:
        # Create a CSV writer object
        csv_writer = csv.writer(csv_file)

        # Write header if needed
        header = ['ID', 'Accuracy']  # Replace with your actual column names
        csv_writer.writerow(header)

        for filename in glob.glob("c2si/lec-all/*-test.json"):
            conf = Confusion(filename)

            patient_id = re.findall(r"-([-\d]+)-", filename)[0].split("-")
            attempt = patient_id[1] if len(patient_id) > 1 else "1"
            patient_id = patient_id[0]

            # TIO-000334-01-L01

            identifier = f"TIO-{patient_id.zfill(6)}-{attempt.zfill(2)}-L01"
            csv_writer.writerow([identifier, conf.balanced])
