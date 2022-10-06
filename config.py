#################################################################################
#                               configuration                                   #
#################################################################################
BATCH_SIZE = 5
EPOCH = 15
start_epoch = 0
learning_rate = 0.001
#################################################################################
#                                    STFT                                      #
#################################################################################
win_len = 512
stride = 128
fft_len = 512
fs = 16000
win_type = 'hann'

#################################################################################
#                                  Data Load                                    #
#################################################################################
path_to_train_noisy = "/home/jbc/TIMIT/TIMIT/train/noisy"
path_to_train_clean = "/home/jbc/TIMIT/TIMIT/train/clean"
path_to_val_noisy = "/home/jbc/TIMIT/TIMIT/valid/noisy"
path_to_val_clean = "/home/jbc/TIMIT/TIMIT/valid/clean"

#################################################################################
#                                    Path                                       #
#################################################################################
import time

file_name = "nunet_lstm_based_1st_v1.h5"
saved_model_dir = './saved_model/'
logdir = 'logs/'
logfile = logdir + file_name
cur_time = time.strftime('%Y-%m-%d-%H-%M-', time.localtime(time.time()))
# logdir = logdir + cur_time + file_name
chpt_dir = 'checkpoint/'
chpt_file = 'checkpoint-{epoch:03d}.ckpt'