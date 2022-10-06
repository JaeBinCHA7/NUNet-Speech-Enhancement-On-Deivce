import numpy as np
import tensorflow as tf
from pesq import pesq as get_pesq
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger, TensorBoard
import config as cfg
from keras.layers import Multiply
from scipy.signal import get_window



###############################################################################
#                              Calculate PESQ                                #
###############################################################################
def cal_pesq(clean_wavs, pred_wavs):
    avg_pesq_score = 0
    for i in range(len(pred_wavs)):
        pesq_score = get_pesq(cfg.fs, clean_wavs[i], pred_wavs[i], "wb")
        avg_pesq_score += pesq_score
    avg_pesq_score /= len(pred_wavs)

    return avg_pesq_score


@tf.function
def pesq(clean_wav, pred_wavs):
    preq_score = tf.numpy_function(cal_pesq, [clean_wav, pred_wavs], tf.float64)

    return preq_score


###############################################################################
#                              Calculate Loss                                #
###############################################################################
@tf.function
def loss_fn(clean_wavs, pred_wavs, r1=1, r2=1):
    r = r1 + r2
    # clean_wavs = tf.signal.frame(clean_wavs, cfg.win_len, cfg.stride)
    # pred_wavs = tf.signal.frame(pred_wavs, cfg.win_len, cfg.stride)

    # clean_mag, _, _ = stft_layer(clean_wavs)
    # pred_mag, _, _ = stft_layer(pred_wavs)
    clean_mags = tf.abs(
        tf.signal.stft(clean_wavs, frame_length=cfg.win_len, frame_step=cfg.stride, fft_length=cfg.fft_len,
                       window_fn=tf.signal.hann_window))
    pred_mags = tf.abs(
        tf.signal.stft(pred_wavs, frame_length=cfg.win_len, frame_step=cfg.stride, fft_length=cfg.fft_len,
                       window_fn=tf.signal.hann_window))

    main_loss = loss_main(clean_wavs, pred_wavs)
    sub_loss = loss_sub(clean_mags, pred_mags)
    loss = (r1 * main_loss + r2 * sub_loss) / r

    return loss


###############################################################################
#                          Audio signal processing                            #
###############################################################################
def stft_layer(x):  # [None, None]
    frames = tf.signal.frame(x, cfg.win_len, cfg.stride)  # [None, None, 512]
    stft_dat = tf.signal.rfft(frames)  # [None, 372, 257]
    mag = tf.abs(stft_dat)  # [None, None, 257]
    phase = tf.math.angle(stft_dat)  # [None, None, 257]

    trans_mags = tf.expand_dims(mag, axis=3)
    trans_mags = trans_mags[:, :, 1:, :]

    return mag, phase, trans_mags


def ifft_layer(x):
    s1_stft = (tf.cast(x[0], tf.complex64) * tf.exp((1j * tf.cast(x[1], tf.complex64))))  # [None, None, 257]
    # return tf.signal.irfft(s1_stft)
    return tf.signal.inverse_stft(s1_stft, cfg.win_len, cfg.stride, fft_length=cfg.fft_len,
                                  window_fn=tf.signal.inverse_stft_window_fn(cfg.stride))


def overlap_add_layer(x):
    return tf.signal.overlap_and_add(x, cfg.stride)


def tf_masking(x, mags):
    out = tf.squeeze(x, axis=3)
    paddings = tf.constant([[0, 0], [0, 0], [1, 0]])  # pad. 3차원 Time 축으로 한칸 제로패딩.
    mask_mags = tf.pad(out, paddings, mode='CONSTANT')  # [1, 372, 257]
    # mask_mags = tf.tanh(mask_mags)  # [None, 483, 257]
    # est_mags = mask_mags * mags  # [None, 483, 257]
    est_mags = Multiply()([mags, mask_mags])

    return est_mags


def spectral_mapping(x):
    out = tf.squeeze(x, axis=3)
    paddings = tf.constant([[0, 0], [0, 0], [1, 0]])  # pad. 3차원 Time 축으로 한칸 제로패딩.
    out = tf.pad(out, paddings, mode='CONSTANT')  # [1, 372, 257, 1] # Direct mapping

    return out


def init_kernels(win_len, fft_len, win_type=None, invers=False):
    if win_type == 'None' or win_type is None:
        window = np.ones(win_len)
    else:
        window = get_window(win_type, win_len, fftbins=True)  # **0.5

    N = fft_len  # 512
    fourier_basis = np.fft.rfft(np.eye(N))[:win_len]  # [400, 257]
    real_kernel = np.real(fourier_basis)  # [400, 257]
    imag_kernel = np.imag(fourier_basis)  # [400, 257]
    kernel = np.concatenate([real_kernel, imag_kernel], 1).T  # [514, 400]

    if invers:
        kernel = np.linalg.pinv(kernel).T

    kernel = (kernel * window).T  # [514, 400]

    kernel = kernel[:, None, :]  # [400, 1, 514]

    return tf.convert_to_tensor(kernel, dtype=tf.float32), tf.convert_to_tensor(
        window[None, None, :], dtype=tf.float32)  # window : [1, 1, 400]


###############################################################################
#                             Define callback                                 #
###############################################################################
tensorboard_callback = TensorBoard(log_dir=cfg.logdir + cfg.file_name[:-3])

model_checkpoint = ModelCheckpoint(filepath=cfg.chpt_dir + cfg.file_name[:-3],
                                   monitor="val_pesq",
                                   save_weights_only=True,
                                   save_best_only=True,
                                   verbose=0,
                                   mode='max',
                                   save_freq='epoch'
                                   )

# create callback for the adaptive learning rate
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, mode="min",
                              patience=3, verbose=1, min_lr=10 ** (-10), cooldown=1)

# create callback for early stopping
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0,
                               patience=10, verbose=1, mode='auto', baseline=None)

# create log file writer
csv_logger = CSVLogger(filename=cfg.logdir + cfg.file_name[:-3] + '.log')
loss_main = tf.keras.losses.MeanSquaredError()
loss_sub = tf.keras.losses.MeanAbsoluteError()
optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate)

callback = []
callback.append(tensorboard_callback)
callback.append(model_checkpoint)
# callback.append(csv_logger)
# callback.append(reduce_lr)
# callback.append(early_stopping)
