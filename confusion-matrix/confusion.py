from sklearn.metrics import confusion_matrix, balanced_accuracy_score, accuracy_score
import matplotlib.pyplot as plt
import itertools
import numpy as np
import json


class Confusion:
    def __init__(self, filepath):
        with open("numeric_phones.json") as f:
            self.numeric_phones = json.load(f)

        with open("phones.json") as f:
            self.phones = json.load(f)

        self.filepath = filepath

        if "bref/" in self.filepath:
            self.labels, self.predicted = self.reader_bref()
        elif "c2si" in self.filepath:
            self.labels, self.predicted = self.reader_c2si()
        else:
            raise Exception("This dataset is not supported yet.")

        self.accuracy, self.balanced = self.get_accuracies()
        self.confusion_matrix = self.compute_confusion_matrix()
        self.plot_confusion_matrix()

    def reader_bref(self):
        def translator(character):
            character = character.replace(" ; ", "").replace(" ", "").replace("<eps>", "")
            index = int(character)

            if index == 0:
                return "silence"

            for phone, i in self.numeric_phones.items():
                if i == index:
                    return phone

        with open(self.filepath) as f:
            lines = [line.replace("\n", "") for line in f.readlines()]

        true_labels = np.array([translator(value) for value in lines[13::5]])
        predicted_labels = np.array([translator(value) for value in lines[15::5]])

        return true_labels, predicted_labels

    def reader_c2si(self):
        with open(self.filepath) as f:
            data = json.load(f)

        def translator(index):
            if index == 0:
                return "silence"

            for phone, i in self.numeric_phones.items():
                if i == index:
                    return phone

        labels = np.array([translator(value[0]) for value in data["labels"]])
        predicted = np.array([translator(value[0]) for value in data["predicted"]])

        return labels, predicted

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
            with open(self.filepath.split('.')[0] + "_accuracies.json", "w+") as f:
                json.dump({
                    "accuracy": round(accuracy*100, 2),
                    "balanced_accuracy": round(balanced_accuracy * 100, 2)
                }, f, indent=4)

        return accuracy, balanced_accuracy

    def compute_confusion_matrix(self):
        label_names_existing = list(set([self.phones[a] for a in self.labels]))
        label_names_organized = ["sil", "a", "Ê", "Û", "Ô", "u", "y", "i", "ã", "ɔ̃", "µ", "w", "ɥ", "j", "l", "ʁ",
                                 "n", "m", "ɲ", "p", "t", "k", "b", "d", "g", "f", "s", "ʃ", "v", "z", "ʒ"]

        cm = confusion_matrix([self.phones[a] for a in self.labels], [self.phones[a] for a in self.predicted], normalize="true")

        # Use the specified labels order to link current cm layout to organized layout
        new_order = [label_names_existing.index(label) for label in label_names_organized if
                     label in label_names_existing]
        cm = cm[:, new_order][new_order, :]

        # Adding lines for phonemes that were not present in the input audio
        for i in range(len(label_names_organized)):
            if label_names_organized[i] not in label_names_existing:
                cm = np.insert(cm, i, 0.0, axis=1)
                cm = np.insert(cm, i, 0.0, axis=0)

        # Set 0 lines to NaN
        for i in range(cm.shape[0]):
            if np.count_nonzero(cm[i, :]) == 0:
                cm[i, :] = np.nan

        return cm

    def plot_confusion_matrix(self, savefig=True, file='', cmap=plt.cm.pink_r):
        label_names_organized_phonetic = ["sil", "a", "Ê", "Û", "Ô", "u", "y", "i", "ã", "ɔ̃", "µ", "w", "ɥ", "j", "l",
                                          "ʁ", "n", "m", "ɲ", "p", "t", "k", "b", "d", "g", "f", "s", "ʃ", "v", "z",
                                          "ʒ"]

        fig, ax = plt.subplots(dpi=125)
        fig.set_size_inches(15, 12, forward=True)

        rect = plt.Rectangle((-0.5, -0.5), self.confusion_matrix.shape[1], self.confusion_matrix.shape[0], linewidth=2, edgecolor='black', facecolor='none')
        ax.add_patch(rect)

        im = ax.imshow(self.confusion_matrix, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
        ax.set_title(file, fontsize=20)
        plt.colorbar(im, ax=ax)

        # TODO Add title to the figure

        tick_marks = np.arange(len(label_names_organized_phonetic))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels([f"/{a}/" for a in label_names_organized_phonetic], fontsize=14)
        ax.set_yticklabels([f"/{a}/" for a in label_names_organized_phonetic], fontsize=14)

        ax.grid(False)

        for i, j in itertools.product(range(self.confusion_matrix.shape[0]), range(self.confusion_matrix.shape[1])):
            color = "white" if self.confusion_matrix[i, j] > 0.6 else "black"
            ax.text(j, i, f'{self.confusion_matrix[i, j] * 100:.1f}',
                    ha="center", va="center", color=color, fontsize=10)

        plt.tight_layout(pad=3)
        ax.set_ylabel('True labels', fontsize=18)
        ax.set_xlabel('Predicted labels', fontsize=18)

        if savefig:
            plt.savefig(self.filepath.split(".")[0] + ".png", bbox_inches='tight', pad_inches=0.1)
            print(f"Confusion matrix was saved at {self.filepath.split('.')[0] + '.png'}")


if __name__ == '__main__':
    # Confusion("bref/cer_test_15_epochs.txt")
    # Confusion("c2si/output_hc_dap.json")
    Confusion("c2si/output_hc_lec.json")
