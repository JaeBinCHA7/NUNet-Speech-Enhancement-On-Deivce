import os, fnmatch
from pathlib import Path
import numpy as np
import soundfile as sf
import config as cfg
import tensorflow as tf
from random import shuffle


# Data Normalization
def minMaxNorm(wav, eps=1e-8):
    max = np.max(abs(wav))
    min = np.min(abs(wav))
    wav = (wav - min) / (max - min + eps)
    return wav

def preprocess_waveform(waveform):
    mn = np.min(waveform)
    mx = np.max(waveform)
    maxabs = np.maximum(np.abs(mn), np.abs(mx))
    # maxabs = np.std(waveform)

    return np.copy(waveform) / maxabs #, (maxabs,)

class audio_generator():
    def __init__(self, path_to_noisy, path_to_clean, train_flag=False, indexing_flag=False, num_dataset=0):
        self.path_to_clean = path_to_clean
        self.path_to_noisy = path_to_noisy
        self.noisy_files_list = fnmatch.filter(os.listdir(path_to_noisy), '*.wav')
        if indexing_flag:
            self.noisy_files_list = self.noisy_files_list[:num_dataset]
        self.num_data = len(self.noisy_files_list)
        self.train_flag = train_flag
        self.create_tf_data_obj()

    def create_generator(self):
        if self.train_flag:
            shuffle(self.noisy_files_list)
        for noisy_file in self.noisy_files_list:
            _lst = []
            for idx, char in enumerate(noisy_file):
                if char == '_':
                    _lst.append(idx)

            clean_file = noisy_file[:_lst[3]]
            clean_file += '.wav'

            noisy_speech, fs = sf.read(os.path.join(self.path_to_noisy, noisy_file))
            clean_speech, fs = sf.read(os.path.join(self.path_to_clean, clean_file))

            noisy_speech, clean_speech = minMaxNorm(noisy_speech), minMaxNorm(clean_speech)
            # noisy_speech, clean_speech = preprocess_waveform(noisy_speech), preprocess_waveform(clean_speech)

            noisy_speech = tf.convert_to_tensor(noisy_speech)
            clean_speech = tf.convert_to_tensor(clean_speech)

            noisy_speech = tf.clip_by_value(noisy_speech, -1, 1)
            clean_speech = tf.clip_by_value(clean_speech, -1, 1)

            # yield noisy_speech.astype("float32"), clean_speech.astype("float32")
            yield noisy_speech, clean_speech


    def create_tf_data_obj(self):
        # creating the tf.data.Dataset from the iterator
        self.tf_data_set = tf.data.Dataset.from_generator(
            self.create_generator,
            (tf.float32, tf.float32),
            output_shapes=(tf.TensorShape([48000]), tf.TensorShape([48000])),
            args=None
        )
        print("# Finished data loading")


print("# Starting data loading")
generator_train = audio_generator(cfg.path_to_train_noisy,
                                  cfg.path_to_train_clean,
                                  train_flag=True,
                                  # indexing_flag=True,
                                  # num_dataset=1000
                                  )
dataset_train = generator_train.tf_data_set
# dataset_train = dataset_train.shuffle(generator_train.total_data // 2)
dataset_train = dataset_train.batch(cfg.BATCH_SIZE, drop_remainder=True).repeat()
# dataset_train = dataset_train.prefetch(tf.data.experimental.AUTOTUNE)
steps_train = generator_train.num_data // cfg.BATCH_SIZE

generator_val = audio_generator(cfg.path_to_val_noisy,
                                cfg.path_to_val_clean,
                                # indexing_flag=True,
                                # num_dataset=100
                                )
dataset_val = generator_val.tf_data_set
dataset_val = dataset_val.batch(cfg.BATCH_SIZE, drop_remainder=True).repeat()
# dataset_val = dataset_val.batch(cfg.BATCH_SIZE, drop_remainder=True)

steps_val = generator_val.num_data // cfg.BATCH_SIZE
