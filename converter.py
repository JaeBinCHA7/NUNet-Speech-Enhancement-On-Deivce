from model import NUNet
from keras import Input, Model
from keras.layers import Concatenate
import tensorflow as tf
import config as cfg


class ConvertTFL():
    def __init__(self, weight_path, tflite_path):
        self.model = NUNet()
        self.model.load_weights(weight_path)
        self.tflite_path = tflite_path

        self.input_layer = self.model.input_layer
        self.encoder_1 = self.model.encoder_1
        self.encoder_2 = self.model.encoder_2
        self.encoder_3 = self.model.encoder_3
        self.encoder_4 = self.model.encoder_4
        self.encoder_5 = self.model.encoder_5
        self.encoder_6 = self.model.encoder_6

        self.ddense = self.model.dense

        self.decoder_1 = self.model.decoder_1
        self.decoder_2 = self.model.decoder_2
        self.decoder_3 = self.model.decoder_3
        self.decoder_4 = self.model.decoder_4
        self.decoder_5 = self.model.decoder_5
        self.decoder_6 = self.model.decoder_6
        self.output_layer = self.model.output_layer

    def converter(self):
        input = Input(batch_shape=(None, 128, 256, 1), ragged=True)
        # input = Input(batch_shape=(None, None, 256, 1))

        # Input Layer
        hx = self.input_layer(input)  # [1, 372, 256, 64]

        # Encoder 1
        en1 = self.encoder_1(hx)  # [1, 372, 128, 64]

        # Encoder 2
        en2 = self.encoder_2(en1)  # [1, 372, 64, 64]

        # Encoder 3
        en3 = self.encoder_3(en2)  # [1, 372, 32, 64]

        # Encoder 4
        en4 = self.encoder_4(en3)  # [1, 372, 16, 64]

        # Encoder 5
        en5 = self.encoder_5(en4)  # [1, 372, 8, 64]

        # Encoder 6
        en6 = self.encoder_6(en5)  # [1, 372, 4, 64]

        # Dilated dense block
        out = self.ddense(en6)  # [1, 372, 4, 64]

        # Decoder 1
        out = self.decoder_1(Concatenate(axis=3)([out, en6]))  # [1, 372, 8, 64]

        # Decoder 2
        out = self.decoder_2(Concatenate(axis=3)([out, en5]))  # [1, 372, 16, 64]

        # Decoder 3
        out = self.decoder_3(Concatenate(axis=3)([out, en4]))  # [1, 372, 32, 64]

        # Decoder 4
        out = self.decoder_4(Concatenate(axis=3)([out, en3]))  # [1, 372, 64, 64]

        # Decoder 5
        out = self.decoder_5(Concatenate(axis=3)([out, en2]))  # [1, 372, 128, 64]

        # Decoder 6
        out = self.decoder_6(Concatenate(axis=3)([out, en1]))  # [1, 372, 256, 64]

        # Output layer
        out = self.output_layer(out)  # [1, 372, 256, 1]

        tflite = Model(inputs=input, outputs=out)
        tflite.summary()

        weigths = self.model.get_weights()
        tflite.set_weights(weigths)

        converter = tf.lite.TFLiteConverter.from_keras_model(tflite)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops. # 필수
            tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
        ]

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.experimental_new_converter = True
        tflite_model = converter.convert()

        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        signatures = interpreter.get_signature_list()
        print(signatures)

        with open(self.tflite_path, 'wb') as f:
            f.write(tflite_model)


###############################################################################
#                         Convert to Tensorflow Lite                          #
###############################################################################
if __name__ == '__main__':
    # weight_path = cfg.chpt_dir + cfg.file_name[:-3]
    weight_path = cfg.chpt_dir + '/nunet_tv4/nunet_tv4'
    m = ConvertTFL(weight_path, "./tflite/nunet_v4_test2.tflite")
    m.converter()
