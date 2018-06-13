import random
import torch.nn.functional as F
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tensorboardX import SummaryWriter
from plotting_utils import plot_alignment_to_numpy, plot_spectrogram_to_numpy
from plotting_utils import plot_gate_outputs_to_numpy
from audio_processing import griffin_lim
from audio_processing import dynamic_range_decompression
import torch
from scipy.io.wavfile import write

def plot_spectrogram(ground_truth, generated_sample, post_net_sample, attention,
 logdir, train_step, number=0, append=False, vmin=None, vmax=None):
  fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, figsize=(8,12))
  
  if vmin is None:
    vmin = min(np.min(ground_truth), np.min(generated_sample), np.min(post_net_sample))
  if vmax is None:
    vmax = max(np.max(ground_truth), np.max(generated_sample), np.min(post_net_sample))
  
  # print(ground_truth.shape)
  # print(generated_sample.shape)

  colour1 = ax1.imshow(ground_truth, cmap='viridis', interpolation='nearest', aspect='auto', vmin=vmin, vmax=vmax)
  colour2 = ax2.imshow(generated_sample, cmap='viridis', interpolation='nearest', aspect='auto', vmin=vmin, vmax=vmax)
  colour3 = ax3.imshow(post_net_sample, cmap='viridis', interpolation='nearest', aspect='auto', vmin=vmin, vmax=vmax)
  colour4 = ax4.imshow(attention.T, cmap='viridis', interpolation='nearest', aspect='auto')
  
  ax1.invert_yaxis()
  ax1.set_ylabel('fourier components')
  ax1.set_title('training data')
  
  ax2.invert_yaxis()
  ax2.set_ylabel('fourier components')
  ax2.set_title('decoder results')

  ax3.invert_yaxis()
  ax3.set_ylabel('fourier components')
  ax3.set_title('post net results')

  ax4.set_title('attention')
  ax4.set_ylabel('inputs')
  
  plt.xlabel('time')
  
  fig.subplots_adjust(right=0.8)
  cbar_ax1 = fig.add_axes([0.85, 0.35, 0.05, 0.5])
  fig.colorbar(colour1, cax=cbar_ax1)
  cbar_ax2 = fig.add_axes([0.85, 0.1, 0.05, 0.15])
  fig.colorbar(colour4, cax=cbar_ax2)
  # fig.colorbar(colour2, ax=ax2)

  if append:
    name = '{}/Output_Step{}_{}_{}.png'.format(logdir, train_step, number, append)
  else:
    name = '{}/Output_Step{}_{}.png'.format(logdir, train_step, number)
  if logdir[0] != '/':
    name = "./"+name
  fig.savefig(name, dpi=300)

  plt.close(fig)

def save_audio(mag_spec, logdir, name, stft, train=True):
  magnitudes = dynamic_range_decompression(mag_spec)
  magnitudes = torch.pow(magnitudes, 1.2)
  magnitudes = torch.unsqueeze(magnitudes, 0)

  # magnitudes = torch.t(magnitudes)
  # print(magnitudes.shape)
  signal = griffin_lim(magnitudes.cpu(), stft.stft_fn)
  signal = signal.data.cpu().numpy()
  if train:
    file_name = '{}/sample_train_step_{}.wav'.format(logdir, name)
  else:
    file_name = '{}/sample_eval_step_{}.wav'.format(logdir, name)
  if logdir[0] != '/':
    file_name = "./"+file_name
  print(signal.shape)
  # signal = signal.astype(np.int16)
  signal = signal[0]
  # print(signal)
  write(file_name, 22050 ,signal)

class Tacotron2Logger(SummaryWriter):
    def __init__(self, logdir):
        super(Tacotron2Logger, self).__init__(logdir)
        self.logdir = logdir

    def log_training(self, reduced_loss, grad_norm, learning_rate, duration,
                     iteration, y, y_pred, stft):
        self.add_scalar("training.loss", reduced_loss, iteration)
        self.add_scalar("grad.norm", grad_norm, iteration)
        self.add_scalar("learning.rate", learning_rate, iteration)
        self.add_scalar("duration", duration, iteration)

        decoder_outputs, mel_outputs, gate_outputs, alignment = y_pred
        mel_targets, gate_targets = y

        idx = random.randint(0, mel_outputs.size(0) - 1)
        # self.add_image(
        #     "alignment",
        #     plot_alignment_to_numpy(alignments[idx].data.cpu().numpy().T),
        #     iteration)
        index = 0

        plot_spectrogram(mel_targets[index].data.cpu().numpy(),
                     decoder_outputs[index].data.cpu().numpy(),
                     mel_outputs[index].data.cpu().numpy(),
                     alignment[index].data.cpu().numpy(),
                     self.logdir, iteration,
                     append="train")

        save_audio(mel_outputs[index], self.logdir, iteration, stft)

    def log_validation(self, reduced_loss, model, y, y_pred, iteration, stft):
        self.add_scalar("validation.loss", reduced_loss, iteration)
        decoder_outputs, mel_outputs, gate_outputs, alignment = y_pred
        mel_targets, gate_targets = y

        # plot distribution of parameters
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            self.add_histogram(tag, value.data.cpu().numpy(), iteration)

        # plot alignment, mel target and predicted, gate target and predicted
        idx = random.randint(0, mel_outputs.size(0) - 1)
        # self.add_image(
        #     "alignment",
        #     plot_alignment_to_numpy(alignments[idx].data.cpu().numpy().T),
        #     iteration)
        index = 0

        plot_spectrogram(mel_targets[index].data.cpu().numpy(),
                     decoder_outputs[index].data.cpu().numpy(),
                     mel_outputs[index].data.cpu().numpy(),
                     alignment[index].data.cpu().numpy(),
                     self.logdir, iteration,
                     append="eval")

        save_audio(mel_outputs[index], self.logdir, iteration, stft, False)

        self.add_image(
            "mel_target",
            plot_spectrogram_to_numpy(mel_targets[idx].data.cpu().numpy()),
            iteration)
        self.add_image(
            "mel_predicted",
            plot_spectrogram_to_numpy(mel_outputs[idx].data.cpu().numpy()),
            iteration)
        self.add_image(
            "gate",
            plot_gate_outputs_to_numpy(
                gate_targets[idx].data.cpu().numpy(),
                F.sigmoid(gate_outputs[idx]).data.cpu().numpy()),
            iteration)
