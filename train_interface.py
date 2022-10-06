import tensorflow as tf
# from model import NUNet
import config as cfg
from tools import callback, optimizer, loss_main, loss_fn, pesq
import time
from dataloader import steps_val, steps_train, dataset_val, dataset_train
from nunet_1st import NUNet
from model_nunet_1st_rt import NUNet_1ST
from model_lstm_based_1st_rt import NUNet_LSTM_1ST
if __name__ == "__main__":
    model = NUNet_LSTM_1ST()
    # model = NUNet()

    model.build(input_shape=(None, 48000))
    model.summary()
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=[pesq], run_eagerly=False)

    print("# Start training original model")
    start = time.time()
    model.load_weights(cfg.chpt_dir + cfg.file_name[:-3])
    # weight_path = cfg.chpt_dir + '/nunet_tv1/nunet_tv1'
    # model.load_weights(weight_path)
    model.fit(dataset_train,
              batch_size=None,
              steps_per_epoch=steps_train,
              epochs=cfg.EPOCH,
              initial_epoch=7,
              verbose=1,
              validation_data=dataset_val,
              validation_steps=steps_val,
              callbacks=callback,
              max_queue_size=50,
              workers=4,
              use_multiprocessing=True
              )

    print("takse {:.2f} seconds".format(time.time() - start))
    model_path = "./saved_model/" + cfg.file_name
    model.save_weights(model_path)
    tf.keras.backend.clear_session()
