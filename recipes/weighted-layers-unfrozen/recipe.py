# From https://colab.research.google.com/drive/17Hu1pxqhfMisjkSgmM2CnZxfqDyn2hSY?usp=sharing#scrollTo=AI6a5KiMQWma

import os
import sys
import torch
import logging
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main, if_main_process
from hyperpyyaml import load_hyperpyyaml
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# class WeightedSSLModel(torch.nn.Module):
#     """This lobe enables the integration of use of weighted sum representations
#     from different layers in a SSL encoder.

#     The model can be used as a fixed feature extractor for SSL benchmarking. It
#     will download automatically the model from HuggingFace or use a local path.

#     More details in recipes/SSL_benchmark

#     Arguments
#     ---------
#     hub : str
#         HuggingFace hub name: e.g "facebook/wav2vec2-large-lv60"
#     num_layers: int
#         Number of internal layers: e.g 13 for "Base" models.
#     layernorm: bool
#         Whether layer representations should be layernormed before sum
#     Example
#     -------
#     >>> inputs = torch.rand([10, ])
#     >>> model_hub = "facebook/wav2vec2-base-h"
#     >>> num_layers = 13
#     >>> model = WeightedSSLModel(model_hub, num_layers)
#     >>> outputs = model(inputs)
#     """

#     def __init__(self, encoder, num_layers, layernorm=False):
#         super().__init__()
#         self.encoder = encoder
#         # self.encoder = AutoModel.from_pretrained(hub, output_hidden_states=True, apply_spec_augment=False)
#         self.num_layers = num_layers
#         # Initializing the learnable weights
#         zero_init = torch.cat([torch.zeros(self.num_layers)])
#         self.weights = torch.nn.Parameter(zero_init, requires_grad=True)
#         self.layernorm = layernorm

#     def forward(self, wav, wav_lens=None):
#         """This method outputs a weighted sum of the layers representations of the SSL encoder
#         Arguments
#         ---------
#         wav : tensor
#             The wavs
#         """

#         feats = self.encoder(wav)
#         # hidden_states = torch.stack(feats.hidden_states, dim=0).detach()
#         hidden_states = feats
#         # First dimension should be equal to the number of layers in the hparams
#         assert (
#             self.num_layers == hidden_states.shape[0]
#         ), "Num layers not equal to num hidden states"
#         norm_weights = torch.nn.functional.softmax(self.weights, dim=-1)
#         # Layernorming the layers representations if asked
#         if self.layernorm:
#             hidden_states = [
#                 F.layer_norm(t, (t.shape[-1],)) for t in hidden_states
#             ]
#         # Summing the weighted layers
#         weighted_feats = hidden_states[0] * norm_weights[0]
#         for i in range(1, len(hidden_states)):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
#             weighted_feats += hidden_states[i] * norm_weights[i]                                                                                                                                                                                                         
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
#         return weighted_feats 

# Define training procedure
class ASR(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)

        feats = self.modules.weighted_ssl_model(wavs)
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
            self.wer_metric.append(ids, predicted, batch.phn_list)  

        return loss

    def fit_batch(self, batch):
        should_step = self.step % self.grad_accumulation_factor == 0

        # Managing automatic mixed precision
        if self.auto_mix_prec:
            # self.wav2vec_optimizer.zero_grad()
            self.model_optimizer.zero_grad()
            self.weights_optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                with self.no_sync():
                    outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            with self.no_sync(not should_step):
                self.scaler.scale(
                    loss / self.grad_accumulation_factor
                ).backward()

            if should_step:
                # if not self.hparams.freeze_wav2vec:
                #     self.scaler.unscale_(self.wav2vec_optimizer)

                self.scaler.unscale_(self.model_optimizer)
                self.scaler.unscale_(self.weights_optimizer)

                if self.check_gradients(loss):
                    # self.scaler.step(self.wav2vec_optimizer)
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
                    # self.wav2vec_optimizer.step()
                    self.model_optimizer.step()
                    self.weights_optimizer.step()

                # self.wav2vec_optimizer.zero_grad()
                self.model_optimizer.zero_grad()
                self.weights_optimizer.zero_grad()
                
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
            stage_stats["WER"] = self.wer_metric.summarize("WER")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr_model, new_lr_model = self.hparams.lr_annealing_model(stage_stats["loss"])
            # old_lr_wav2vec, new_lr_wav2vec = self.hparams.lr_annealing_wav2vec(stage_stats["loss"])

            sb.nnet.schedulers.update_learning_rate(self.model_optimizer, new_lr_model)
            # sb.nnet.schedulers.update_learning_rate(self.wav2vec_optimizer, new_lr_wav2vec)

            old_lr_encoder, new_lr_encoder = self.hparams.lr_annealing_weights(stage_stats["loss"])
            sb.nnet.schedulers.update_learning_rate(self.weights_optimizer, new_lr_encoder)

            print(self.modules.weighted_ssl_model.weights)

            self.hparams.train_logger.log_stats(
                stats_meta={
                    "epoch": epoch,
                    "lr_model": old_lr_model,
                    # "lr_wav2vec": old_lr_wav2vec,
                    "lr_weights": old_lr_encoder
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
        # Handling SpeechBrain vs HuggingFace pretrained models
        # if hasattr(self.modules, "extractor"):  # SpeechBrain pretrained model
        #     self.wav2vec_optimizer = self.hparams.wav2vec_opt_class(
        #         self.modules.encoder_wrapper.parameters()
        #     )

        # else:  # HuggingFace pretrained model
        #     self.wav2vec_optimizer = self.hparams.wav2vec_opt_class(
        #         self.modules.wav2vec2.parameters()
        #     )

        self.weights_optimizer = self.hparams.weights_opt_class(
            [self.modules.weighted_ssl_model.weights]
        )

        self.model_optimizer = self.hparams.model_opt_class(
            self.hparams.model.parameters()
        )

        if self.checkpointer is not None:
            # self.checkpointer.add_recoverable("wav2vec_opt", self.wav2vec_optimizer)
            self.checkpointer.add_recoverable("modelopt", self.model_optimizer)
            self.checkpointer.add_recoverable("weights_opt", self.weights_optimizer)

    def zero_grad(self, set_to_none=False):
        # self.wav2vec_optimizer.zero_grad(set_to_none)
        self.model_optimizer.zero_grad(set_to_none)
        self.weights_optimizer.zero_grad(set_to_none)


def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"], replacements={"data_folder": data_folder},
    )

    # we sort training data to speed up training and get better results.
    train_data = train_data.filtered_sorted(sort_key="wav")
    hparams["train_dataloader_opts"]["shuffle"] = False

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"], replacements={"data_folder": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="wav")

    # test is separate
    test_dataset = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_csv"], replacements={"data_folder": data_folder},
    )
    test_dataset = test_dataset.filtered_sorted(sort_key="wav")

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


if __name__ == "__main__":

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # here we create the datasets objects as well as tokenization and encoding
    train_data, valid_data, test_dataset = dataio_prepare(
        hparams
    )

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

    # Training
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    # Testing
    if not os.path.exists(hparams["output_wer_folder"]):
        os.makedirs(hparams["output_wer_folder"])

    asr_brain.hparams.test_wer_file = os.path.join(hparams["output_wer_folder"], "wer_test.txt")

    asr_brain.evaluate(
        test_dataset,
        test_loader_kwargs=hparams["test_dataloader_opts"],
        min_key="WER",
    )
