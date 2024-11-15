import re

from sklearn.metrics import confusion_matrix, balanced_accuracy_score, accuracy_score
import matplotlib.pyplot as plt
import itertools
import numpy as np
import json


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

        print(len(data["labels"]))

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
        self.output_png = self.filepath.split(".")[0] + output_suffix + ".pdf"
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
        self.confusion_matrix = self.compute_confusion_matrix()
        self.plot_confusion_matrix()

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
        # print(self.confusion_matrix)
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

        fontsize = linear_mapping_function(1, 37, 31, 14)(len(labels))
        ax.set_xticklabels([f"/{a}/" for a in labels], fontsize=fontsize, weight='bold')
        ax.set_yticklabels([f"/{a}/" for a in labels], fontsize=fontsize, weight='bold')

        ax.grid(False)

        fontsize = linear_mapping_function(1, 30, 31, 10)(len(labels))
        for i, j in itertools.product(range(self.confusion_matrix.shape[0]), range(self.confusion_matrix.shape[1])):
            color = "white" if self.confusion_matrix[i, j] > 0.6 else "black"
            ax.text(j, i, f'{self.confusion_matrix[i, j] * 100:.1f}',
                    ha="center", va="center", color=color, fontsize=fontsize, weight='bold')

        plt.tight_layout(pad=3)
        ax.set_ylabel('Ground truth', fontsize=30, weight='bold')
        ax.set_xlabel('Predicted labels', fontsize=30, weight='bold')

        if savefig:
            plt.savefig(self.output_png, bbox_inches='tight', pad_inches=0.1)
            print(f"Confusion matrix was saved at {self.output_png}")


def launch(folder, *args):
    if 'bref' in args:
        Confusion(f"bref/{folder}/output_test.json")
    if 'dap' in args:
        Confusion(f"c2si/{folder}/output_hc_dap.json")
    if 'lec' in args:
        Confusion(f"c2si/{folder}/output_hc_lec.json")
    if 'hc' in args:
        Confusion(f"c2si/{folder}/output_healthy_controls.json")
    if 'patients' in args:
        Confusion(f"c2si/{folder}/output_patients_dap.json")
        Confusion(f"c2si/{folder}/output_patients_lec.json")

    if 'oral-nasal' in args:
        Confusion(f"c2si/{folder}/output_hc_lec.json", ["a", "Ê", "Û", "Ô", "u", "y", "i", "ã", "ɔ̃", "µ", "n", "m"], "_oral_nasal")
        Confusion(f"bref/{folder}/output_test.json", ["a", "Ê", "Û", "Ô", "u", "y", "i", "ã", "ɔ̃", "µ", "n", "m"], "_oral_nasal")

    if 'obstruent' in args:
        Confusion(f"c2si/{folder}/output_hc_lec.json", ["p", "t", "k", "b", "d", "g", "f", "s", "ʃ", "v", "z", "ʒ"], "_obstruent")
        Confusion(f"bref/{folder}/output_test.json", ["p", "t", "k", "b", "d", "g", "f", "s", "ʃ", "v", "z", "ʒ"], "_obstruent")


if __name__ == '__main__':
    # launch('frozen',      'obstruent')  # FROZEN
    # launch('frozen', 'bref', 'dap', 'lec', 'hc', 'oral-nasal', 'obstruent')  # FROZEN
    # launch('unfrozen', 'bref', 'dap', 'lec', 'oral-nasal', 'obstruent')  # UNFROZEN
    # launch('weights-global-training', 'bref', 'dap', 'lec', 'oral-nasal', 'obstruent')  # WEIGHTS
    # launch('unfrozen-loss', 'bref', 'dap', 'lec', 'oral-nasal', 'obstruent')  # UNFROZEN - WITH MINIMAL VALIDATION LOSS
    # launch('weights-baseline', 'bref', 'dap', 'lec', 'oral-nasal', 'obstruent')  # WEIGHTS - BASELINE (UNIFORM WEIGHTS DISTRIBUTION)
    # launch('weights-unfrozen', 'bref', 'dap', 'lec', 'oral-nasal', 'obstruent')  # WEIGHTS - FROM UNFROZEN
    # launch('weights-frozen', 'bref', 'dap', 'lec', 'oral-nasal', 'obstruent')  # WEIGHTS - FROM UNFROZEN
    # launch('unfrozen-3k-large', 'bref', 'dap', 'lec', 'oral-nasal', 'obstruent')  # WEIGHTS - FROM UNFROZEN
    # launch('unfrozen-3k-base', 'bref', 'dap', 'lec', 'oral-nasal', 'obstruent')  # WEIGHTS - FROM UNFROZEN

    # launch('unfrozen-cp-3k-base', 'bref', 'dap', 'lec', 'oral-nasal', 'obstruent')  # WEIGHTS - FROM UNFROZEN
    # launch('unfrozen-cp-3k-large', 'bref', 'dap', 'lec', 'oral-nasal', 'obstruent')  # WEIGHTS - FROM UNFROZEN
    # launch('unfrozen-cp-14k-light', 'bref', 'dap', 'lec', 'oral-nasal', 'obstruent')  # WEIGHTS - FROM UNFROZEN
    # launch('unfrozen-cp-14k-large', 'bref', 'dap', 'lec', 'oral-nasal', 'obstruent')  # WEIGHTS - FROM UNFROZEN
    # launch('unfrozen-cp-1k', 'bref', 'dap', 'lec', 'oral-nasal', 'obstruent')  # WEIGHTS - FROM UNFROZEN
    # launch('unfrozen-1k', 'bref', 'dap', 'lec', 'oral-nasal', 'obstruent')  # WEIGHTS - FROM UNFROZEN

    # launch('unfrozen-14k-light', 'bref', 'dap', 'lec', 'oral-nasal', 'obstruent')  # WEIGHTS - FROM UNFROZEN

    # for i in range(1, 11):
    #     launch(f'layers/{i}', 'bref', 'dap', 'lec')

    # launch('unfrozen-cp-3k-large-accents', 'bref', 'dap', 'lec', 'oral-nasal', 'obstruent', "patients")  # WEIGHTS - FROM UNFROZEN
    # launch('unfrozen-cp-3k-large-accents', 'bref', 'dap', 'lec', 'oral-nasal', 'obstruent')  # WEIGHTS - FROM UNFROZEN
    # launch('unfrozen-cp-3k-base-accents', 'bref', 'dap', 'lec', 'oral-nasal', 'obstruent')  # WEIGHTS - FROM UNFROZEN
    # launch('unfrozen-cp-14k-large-accents', 'bref', 'dap', 'lec', 'oral-nasal', 'obstruent')  # WEIGHTS - FROM UNFROZEN
    # launch('unfrozen-cp-14k-light-accents', 'bref', 'dap', 'lec', 'oral-nasal', 'obstruent')  # WEIGHTS - FROM UNFROZEN

    # launch('unfrozen-cp-lv-60-accents', 'bref', 'dap', 'lec', 'oral-nasal', 'obstruent')
    # launch('unfrozen-cp-xlsr-53-accents', 'bref', 'dap', 'lec', 'oral-nasal', 'obstruent')

    # launch('relu-snn-data-test', 'bref')
    # launch('relu-snn-data-train', 'bref')
    # launch('relu-snn-data-valid', 'bref')

    # launch('brefint-30', 'bref')

    # launch('best-relu-test', 'bref', 'dap', 'lec', 'oral-nasal', 'obstruent', "patients")
    # launch('best-relu-long-context', 'bref', 'dap', 'lec', 'oral-nasal', 'obstruent', 'patients')
    # launch('best-relu-middle-segment', 'bref', 'dap', 'lec', 'oral-nasal', 'obstruent', 'patients')

    launch('best-relu-middle-segment-dropout', 'bref', 'dap', 'lec', 'oral-nasal', 'obstruent', 'patients')
    launch('best-relu-longer-context-dropout', 'bref', 'dap', 'lec', 'oral-nasal', 'obstruent', 'patients')
