import soundfile as sf
import numpy as np
import tensorflow as tf
import time
from pesq import pesq as get_pesq
import config as cfg
from tools import stft_layer

# load models
interpreter = tf.lite.Interpreter(model_path='./tflite/nunet_v4_test.tflite')

interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("TFLite input", input_details[0]['shape'])
print("TFLite output", output_details[0]['shape'])

audio, fs = sf.read('./dataset/noisy3.wav')
audio = np.expand_dims(audio, axis=0)
if fs != 16000:
    raise ValueError('This model only supports 16k sampling rate.')
start_time = time.time()
# in_mag, in_phase, sliced_mag = stft_layer(audio)
frames = tf.signal.stft(audio, frame_length=cfg.win_len, frame_step=cfg.stride, fft_length=cfg.fft_len,
                        window_fn=tf.signal.hann_window)
mags = tf.abs(frames)  # [None, None, 257]
phase = tf.math.angle(frames)  # [None, None, 257]

trans_mags = tf.expand_dims(mags, axis=3)
trans_mags = trans_mags[:, :, 1:, :]
trans_mags = tf.cast(trans_mags, dtype=tf.float32)

interpreter.set_tensor(input_details[0]['index'], trans_mags)
interpreter.invoke()
tflite_out = interpreter.get_tensor(output_details[0]['index'])

out_mask = np.squeeze(tflite_out, axis=3)  # [1, 1, 256]
paddings = np.array([[0, 0], [0, 0], [1, 0]])  # pad. 3차원 Time 축으로 한칸 제로패딩.
out_mask = np.pad(out_mask, paddings)  # [1, 1, 257]

estimated_complex = (tf.cast(mags * out_mask, tf.complex64) *
                     tf.exp((1j * tf.cast(phase, tf.complex64))))  # [None, None, 257]
# estimated_block = tf.signal.irfft(estimated_complex)
enhanced = tf.signal.inverse_stft(estimated_complex, cfg.win_len, cfg.stride, fft_length=cfg.fft_len,
                                         window_fn=tf.signal.inverse_stft_window_fn(cfg.stride))
# enhanced = tf.signal.overlap_and_add(estimated_block, 128)

out_file = np.squeeze(enhanced)
print('Processing Time [ms]:', time.time() - start_time)

sf.write('./dataset/enhanced3_tflite.wav', out_file, fs)

print('Processing finished.')

enhanced, fs = sf.read("./dataset/enhanced3_tflite.wav")
clean, fs = sf.read("./dataset/clean3.wav")
noisy, fs = sf.read("./dataset/noisy3.wav")


def cal_pesq(y_true, y_pred):
    sr = 16000
    mode = "wb"
    pesq_score = get_pesq(sr, y_true, y_pred, mode)
    # pesq_score = get_pesq(sr, enhanced, clean, mode)
    return pesq_score


print("PESQ : ", cal_pesq(clean, enhanced))
print("Noisy audio : ", cal_pesq(clean, noisy))
