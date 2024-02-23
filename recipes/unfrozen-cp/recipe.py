# From https://colab.research.google.com/drive/17Hu1pxqhfMisjkSgmM2CnZxfqDyn2hSY?usp=sharing#scrollTo=AI6a5KiMQWma

import os
import sys
import torch
import logging
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main, if_main_process
from hyperpyyaml import load_hyperpyyaml
import torch.nn.functional as F

from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import json

logger = logging.getLogger(__name__)


# Define training procedure
class ASR(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)

        # Forward pass (NOTE: I removed some code here)

        # Handling SpeechBrain vs HuggingFace pretrained models
        if hasattr(self.modules, "extractor"):  # SpeechBrain pretrained model
            latents = self.modules.extractor(wavs)
            feats = self.modules.encoder_wrapper(latents, wav_lens=wav_lens)[
                "embeddings"
            ]
        else:  # HuggingFace pretrained model
            feats = self.modules.wav2vec2(wavs, wav_lens)

        x = self.modules.enc(feats.view(feats.size(0), -1))

        # Compute outputs
        logits = self.modules.ctc_lin(x)

        return logits, wav_lens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC+NLL) given predictions and targets."""
        logits, _ = predictions
        ids = batch.id
        tokens, _ = batch.phn_encoded

        target = torch.zeros(logits.size(0), logits.size(1), dtype=torch.float)
        target = target.to(self.device)
        target.scatter_(1, tokens, 1)

        loss = F.cross_entropy(logits, target)

        if stage != sb.Stage.TRAIN:
            # Computing phoneme error rate
            predicted = logits.max(1).indices.view(logits.size(0), 1)
            predicted = [[str(element) for element in sublist] for sublist in predicted.tolist()]

            # predicted = predicted.cpu().detach().numpy().astype(np.dtype.str).tolist()
            self.wer_metric.append(ids, predicted, batch.phn_list)  

        return loss

    def fit_batch(self, batch):
        should_step = self.step % self.grad_accumulation_factor == 0

        # Managing automatic mixed precision
        if self.auto_mix_prec:
            self.wav2vec_optimizer.zero_grad()
            self.model_optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                with self.no_sync():
                    outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            with self.no_sync(not should_step):
                self.scaler.scale(
                    loss / self.grad_accumulation_factor
                ).backward()
            if should_step:
                if not self.hparams.freeze_wav2vec:
                    self.scaler.unscale_(self.wav2vec_optimizer)
                self.scaler.unscale_(self.model_optimizer)
                if self.check_gradients(loss):
                    self.scaler.step(self.wav2vec_optimizer)
                    self.scaler.step(self.model_optimizer)
                self.scaler.update()
                self.optimizer_step += 1
        else:
            with self.no_sync():
                outputs = self.compute_forward(batch, sb.Stage.TRAIN)
            loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            (loss / self.grad_accumulation_factor).backward()
            if should_step:
                if self.check_gradients(loss):
                    self.wav2vec_optimizer.step()
                    self.model_optimizer.step()
                self.wav2vec_optimizer.zero_grad()
                self.model_optimizer.zero_grad()
                self.optimizer_step += 1

        return loss.detach().cpu()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            # self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            # stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr_model, new_lr_model = self.hparams.lr_annealing_model(
                stage_stats["loss"]
            )
            old_lr_wav2vec, new_lr_wav2vec = self.hparams.lr_annealing_wav2vec(
                stage_stats["loss"]
            )
            sb.nnet.schedulers.update_learning_rate(
                self.model_optimizer, new_lr_model
            )
            sb.nnet.schedulers.update_learning_rate(
                self.wav2vec_optimizer, new_lr_wav2vec
            )
            self.hparams.train_logger.log_stats(
                stats_meta={
                    "epoch": epoch,
                    "lr_model": old_lr_model,
                    "lr_wav2vec": old_lr_wav2vec,
                },
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"WER": stage_stats["WER"]}, min_keys=["WER"],
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            if if_main_process():
                with open(self.hparams.test_wer_file, "w") as w:
                    self.wer_metric.write_stats(w)

    def init_optimizers(self):
        "Initializes the wav2vec2 optimizer and model optimizer"
        # Handling SpeechBrain vs HuggingFance pretrained models
        if hasattr(self.modules, "extractor"):  # SpeechBrain pretrained model
            self.wav2vec_optimizer = self.hparams.wav2vec_opt_class(
                self.modules.encoder_wrapper.parameters()
            )

        else:  # HuggingFace pretrained model
            self.wav2vec_optimizer = self.hparams.wav2vec_opt_class(
                self.modules.wav2vec2.parameters()
            )

        self.model_optimizer = self.hparams.model_opt_class(
            self.hparams.model.parameters()
        )

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable(
                "wav2vec_opt", self.wav2vec_optimizer
            )
            self.checkpointer.add_recoverable("modelopt", self.model_optimizer)

    def zero_grad(self, set_to_none=False):
        self.wav2vec_optimizer.zero_grad(set_to_none)
        self.model_optimizer.zero_grad(set_to_none)

    def transcribe_dataset(self, dataset, min_key, loader_kwargs):
        # If dataset isn't a Dataloader, we create it. 
        if not isinstance(dataset, DataLoader):
            loader_kwargs["ckpt_prefix"] = None
            dataset = self.make_dataloader(
                dataset, sb.Stage.TEST, **loader_kwargs
            )

        self.on_evaluate_start(min_key=min_key) # We call the on_evaluate_start that will load the best model
        self.modules.eval() # We set the model to eval mode (remove dropout etc)

        # Now we iterate over the dataset and we simply compute_forward and decode
        with torch.no_grad():

            transcripts = []
            truth = []
            for batch in tqdm(dataset, dynamic_ncols=True):
                # Make sure that your compute_forward returns the predictions !!!
                logits, _ = self.compute_forward(batch, stage=sb.Stage.TEST) 
                predicted = logits.max(1).indices.view(logits.size(0), 1).tolist()
                
                transcripts += predicted
                truth += [[int(b) for b in a] for a in batch.phn_list]

        return transcripts, truth
    

def load_dataset(name: str, replacement=None):
    output = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams[name], replacements=replacement,
    )

    # we sort training data to speed up training and get better results.
    return output.filtered_sorted(sort_key="wav")


def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    
    hparams["train_dataloader_opts"]["shuffle"] = False
    data_folder, data_folder_cp = hparams["data_folder"], hparams["data_folder_cp"]

    train_data = load_dataset("train_csv", {"data_folder": data_folder, "cp_data_folder": data_folder_cp})
    valid_data = load_dataset("valid_csv", {"data_folder": data_folder, "cp_data_folder": data_folder_cp})
    test_dataset = load_dataset("test_csv", {"data_folder": data_folder, "cp_data_folder": data_folder_cp})

    datasets = [train_data, valid_data, test_dataset]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav", "start")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav, start):
        # TODO Also care about what happens if the segment is located at the end of an audio!
        fr: int = int(hparams["sample_rate"] / 1_000)
        start = max(0, int(start) * fr - (hparams["segment_length"] - 10) // 2)
        stop = start + hparams["segment_length"]

        sig = sb.dataio.dataio.read_audio(({
            "file": wav,
            "start": start,
            "stop": stop
        }))
    
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("phn")
    @sb.utils.data_pipeline.provides("phn_list", "phn_encoded")
    def text_pipeline(phn):
        yield [phn]
        yield torch.LongTensor([int(phn)])

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "sig", "phn_list", "phn_encoded"],
    )

    return train_data, valid_data, test_dataset


def dataio_prepare_transcript(hparams, hdatasets):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""

    transcription_dataset = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hdatasets["transcription_dataset"], replacements={"data_folder": hdatasets["data_folder"]},
    )
    transcription_dataset = transcription_dataset.filtered_sorted(sort_key="wav")

    datasets = [transcription_dataset]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav", "start")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav, start):
        # TODO Also care about what happens if the segment is located at the end of an audio!
        fr: int = int(hparams["sample_rate"] / 1_000)
        start = max(0, int(start) * fr - (hparams["segment_length"] - 10) // 2)
        stop = start + hparams["segment_length"] + 1

        sig = sb.dataio.dataio.read_audio(({
            "file": wav,
            "start": start,
            "stop": stop
        }))
    
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("phn")
    @sb.utils.data_pipeline.provides("phn_list", "phn_encoded")
    def text_pipeline(phn):
        yield [phn]
        yield torch.LongTensor([int(phn)])

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "sig", "phn_list", "phn_encoded"],
    )

    return transcription_dataset


if __name__ == "__main__":

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    # sb.create_experiment_directory(
    #     experiment_directory=hparams["output_folder"],
    #     hyperparams_to_save=hparams_file,
    #     overrides=overrides,
    # )

    # here we create the datasets objects as well as tokenization and encoding
    # train_data, valid_data, test_dataset = dataio_prepare(
    #     hparams
    # )

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # We load the pretrained wav2vec2 model
    if "pretrainer" in hparams.keys():
        run_on_main(hparams["pretrainer"].collect_files)
        hparams["pretrainer"].load_collected(asr_brain.device)

    # We dynamicaly add the tokenizer to our brain class.
    # NB: This tokenizer corresponds to the one used for the LM!!
    # asr_brain.tokenizer = label_encoder

    # Training
    # asr_brain.fit(
    #     asr_brain.hparams.epoch_counter,
    #     train_data,
    #     valid_data,
    #     train_loader_kwargs=hparams["train_dataloader_opts"],
    #     valid_loader_kwargs=hparams["valid_dataloader_opts"],
    # )

    # Testing
    # if not os.path.exists(hparams["output_wer_folder"]):
    #     os.makedirs(hparams["output_wer_folder"])

    # asr_brain.hparams.test_wer_file = os.path.join(hparams["output_wer_folder"], "wer_test.txt")

    # asr_brain.evaluate(
    #     test_dataset,
    #     test_loader_kwargs=hparams["test_dataloader_opts"],
    #     min_key="WER",
    # )

    for k, v in hparams["to_transcribe"].items():
        transcription_dataset = dataio_prepare_transcript(hparams, v)
        
        transcripts, truth = asr_brain.transcribe_dataset(
            dataset=transcription_dataset,  # Must be obtained from the dataio_function
            min_key="WER",  # We load the model with the lowest WER
            loader_kwargs=hparams["transcribe_dataloader_opts"], # opts for the dataloading
        )

        with open(hparams["output_transcription"].format(dataset=k), "w+") as f:
            json.dump({
                "labels": truth,
                "predicted": transcripts
            }, f)
