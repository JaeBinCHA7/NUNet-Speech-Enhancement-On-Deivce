import keras
import tensorflow as tf
from keras.layers import Conv2D, Conv2DTranspose, LayerNormalization, PReLU, Lambda, Concatenate, Reshape, Permute, \
    ZeroPadding2D
from keras.regularizers import l2
from tools import ifft_layer, tf_masking


# Causal Convolution
class causalConv2d(tf.Module):
    def __init__(self, out_ch, kernel_size, stride=1, padding=1, dilation=1, groups=1, name=None):
        super(causalConv2d, self).__init__(name=name)
        self.conv = Conv2D(out_ch, kernel_size=kernel_size, strides=stride, padding='valid', dilation_rate=dilation,
                           groups=groups)
        # self.padding = tf.constant([[0, 0], [padding[0], 0], [padding[1] // 2, padding[1] // 2], [0, 0]])
        self.padding_layer = ZeroPadding2D(padding=((padding[0], 0), (padding[1] // 2, padding[1] // 2)))

    def __call__(self, x):
        # x = tf.pad(x, self.padding)
        x = self.padding_layer(x)
        x = self.conv(x)

        return x


# Convolution block
class CONV(tf.Module):
    def __init__(self, out_ch, name=None):
        super(CONV, self).__init__(name=name)
        self.conv = causalConv2d(out_ch, kernel_size=(2, 3), stride=(1, 2), padding=(1, 2), name='causalConv2d')
        self.ln = LayerNormalization(epsilon=1e-8)
        self.prelu = PReLU(alpha_initializer=tf.initializers.constant(0.25), shared_axes=[1, 2, 3])

    def __call__(self, x):
        return self.prelu(self.ln(self.conv(x)))


# Convolution block for input layer
class Inconv(tf.Module):
    def __init__(self, out_ch, name=None):
        super(Inconv, self).__init__(name=name)
        self.conv = Conv2D(out_ch, kernel_size=1)
        self.ln = LayerNormalization(epsilon=1e-8)
        self.prelu = PReLU(alpha_initializer=tf.initializers.constant(0.25), shared_axes=[1, 2, 3])

    def __call__(self, x):
        return self.prelu(self.ln(self.conv(x)))


# Sub-pixel Convolution Layer
class Spconv(tf.Module):
    def __init__(self, out_ch, scale_factor=2, name=None):
        super(Spconv, self).__init__(name=name)
        self.CONV = causalConv2d(out_ch * scale_factor, kernel_size=(2, 3), padding=(1, 2), name='causalConv2d')
        self.ln = LayerNormalization(epsilon=1e-8)
        self.prelu = PReLU(alpha_initializer=tf.initializers.constant(0.25), shared_axes=[1, 2, 3])
        self.n = scale_factor

    def __call__(self, x):  # [B, T, F, C]
        x = self.CONV(x)
        x = Reshape([-1, x.shape[2], x.shape[3] // self.n, self.n])(x)  # [B, T, F, C//2 , 2]
        x = Permute((1, 2, 4, 3))(x)  # [B, T, F, 2, C//2]
        x = Reshape([-1, x.shape[2] * self.n, x.shape[4]])(x)  # [B, T, 2*F, C//2]

        x = self.ln(x)
        x = self.prelu(x)

        return x


# 1x1 CONV for down-sampling
class DownSampling(tf.Module):
    def __init__(self, in_ch, name=None):
        super(DownSampling, self).__init__(name=name)
        self.down_sampling = Conv2D(in_ch, kernel_size=(1, 3), strides=(1, 2), padding='same')

    def __call__(self, x):
        return self.down_sampling(x)


# 1x1 CONV for up-sampling
class UpSampling(tf.Module):
    def __init__(self, in_ch, name=None):
        super(UpSampling, self).__init__(name=name)
        self.up_sampling = Conv2DTranspose(in_ch, kernel_size=(1, 3), strides=(1, 2), padding='same')

    def __call__(self, x):  # [1, 372, 4, 128]
        out = self.up_sampling(x)
        return out


class dilated_dense_block(tf.Module):
    def __init__(self, in_ch, out_ch, n_layers, l2_regularization=0.01, name=None):
        super(dilated_dense_block, self).__init__(name=name)
        self.input_layer = causalConv2d(in_ch // 2, kernel_size=(2, 3), padding=(1, 2), name='causalConv2d')
        self.prelu1 = PReLU(alpha_initializer=tf.initializers.constant(0.25), shared_axes=[1, 2, 3])
        self.regularization = l2(l2_regularization)

        self.layers = []
        for i in range(n_layers):
            self.layers.append(keras.Sequential(
                [
                    # causalConv2d(in_ch // 2, kernel_size=(2, 3), padding=(2 ** i, 2 ** (i + 1)), dilation=2 ** i,
                    #              groups=in_ch // 2, name='depth_wise_convolution'),
                    # Conv2D(in_ch // 2, kernel_size=1),
                    causalConv2d((in_ch // 2) + i * in_ch // 2, kernel_size=(2, 3), padding=(2 ** i, 2 ** (i + 1)),
                                 dilation=2 ** i,
                                 groups=(in_ch // 2) + i * in_ch // 2, name='depth_wise_convolution'),
                    Conv2D(in_ch // 2, kernel_size=1),
                    # ZeroPadding2D(padding=((2 ** i, 0), (2 ** (i + 1), 0))),
                    # SeparableConv2D(in_ch // 2, kernel_size=(2, 3), padding='same',
                    #                 kernel_regularizer=self.regularization, dilation_rate=(2 ** i)),
                    LayerNormalization(epsilon=1e-8),
                    PReLU(alpha_initializer=tf.initializers.constant(0.25), shared_axes=[1, 2, 3])
                ]
            ))

        self.output_layer = causalConv2d(out_ch, kernel_size=(2, 3), padding=(1, 2), name='causalConv2d')
        self.prelu2 = PReLU(alpha_initializer=tf.initializers.constant(0.25), shared_axes=[1, 2, 3])

    def __call__(self, x):
        x = self.input_layer(x)
        x = self.prelu1(x)  # [1, 372, 4, 16]

        out1 = self.layers[0](x)  # [1, 372, 4, 16]

        out2 = Concatenate(axis=3)([out1, x])  # [1, 372, 4, 32]
        out2 = self.layers[1](out2)  # [1, 372, 4, 16]

        out3 = Concatenate(axis=3)([out2, out1])  # [1, 372, 4, 32]
        out3 = Concatenate(axis=3)([out3, x])  # [1, 372, 4, 48]
        out3 = self.layers[2](out3)  # [1, 372, 4, 16]

        out4 = Concatenate(axis=3)([out3, out2])  # [1, 372, 4, 32]
        out4 = Concatenate(axis=3)([out4, out1])  # [1, 372, 4, 48]
        out4 = Concatenate(axis=3)([out4, x])  # [1, 372, 4, 64]
        out4 = self.layers[3](out4)  # [1, 372, 4, 16]

        out5 = Concatenate(axis=3)([out4, out3])
        out5 = Concatenate(axis=3)([out5, out2])
        out5 = Concatenate(axis=3)([out5, out1])
        out5 = Concatenate(axis=3)([out5, x])  # [1, 372, 4, 80]
        out5 = self.layers[4](out5)  # [1, 372, 4, 16]

        out = Concatenate(axis=3)([out5, out4])
        out = Concatenate(axis=3)([out, out3])
        out = Concatenate(axis=3)([out, out2])
        out = Concatenate(axis=3)([out, out1])
        out = Concatenate(axis=3)([out, x])  # [1, 372, 4, 96]
        out = self.layers[5](out)  # [1, 372, 4, 16]

        out = self.output_layer(out)
        out = self.prelu2(out)

        return out


# Multi-Scale Feature Extraction (MSFE)
class MSFE(tf.Module):
    def __init__(self, iter, mid_ch, out_ch, name=None):
        super(MSFE, self).__init__(name=name)
        self.iter = iter
        self.dense = dilated_dense_block(mid_ch, mid_ch, 6, name="DDense")
        self.Inconv = Inconv(out_ch, name="Inconv")
        self.encoder_list = []
        self.decoder_list = []
        for i in range(self.iter):
            self.encoder_list.append(CONV(mid_ch, name="CONV"))
            if i == self.iter - 1:
                self.decoder_list.append(Spconv(out_ch, name="Spconv"))
            else:
                self.decoder_list.append(Spconv(mid_ch, name="Spconv"))

    def __call__(self, input):
        input = self.Inconv(input)
        x = input

        # Encdoer
        encoder_out = []
        for i in range(self.iter):
            x = self.encoder_list[i](x)
            encoder_out.append(x)

        # Bottleneck
        x = self.dense(x)  # [1, 372, 4, 32]

        # Decoder
        for i in range(self.iter):
            x = Concatenate(axis=3)([x, encoder_out[-i - 1]])
            if i == self.iter - 1:
                x = self.decoder_list[i](x)
                x += input

                return x
            x = self.decoder_list[i](x)


###################################################################################################################3

# Multi-Scale Feature Extraction (MSFE) - 6
class MSFE6(tf.Module):
    def __init__(self, mid_ch, out_ch, name=None):
        super(MSFE6, self).__init__(name=name)
        self.input_layer = Inconv(out_ch)

        # encoder
        self.en1 = CONV(mid_ch)
        self.en2 = CONV(mid_ch)
        self.en3 = CONV(mid_ch)
        self.en4 = CONV(mid_ch)
        self.en5 = CONV(mid_ch)
        self.en6 = CONV(mid_ch)

        # bottleneck
        self.ddense = dilated_dense_block(mid_ch, mid_ch, 6)

        # decoder
        self.de1 = Spconv(mid_ch)
        self.de2 = Spconv(mid_ch)
        self.de3 = Spconv(mid_ch)
        self.de4 = Spconv(mid_ch)
        self.de5 = Spconv(mid_ch)
        self.de6 = Spconv(out_ch)

    def __call__(self, x):
        x = self.input_layer(x)

        # encoder
        out1 = self.en1(x)
        out2 = self.en2(out1)
        out3 = self.en3(out2)
        out4 = self.en4(out3)
        out5 = self.en5(out4)
        out6 = self.en6(out5)

        # bottleneck
        out = self.ddense(out6)

        # decoder
        out = self.de1(Concatenate(axis=3)([out, out6]))
        out = self.de2(Concatenate(axis=3)([out, out5]))
        out = self.de3(Concatenate(axis=3)([out, out4]))
        out = self.de4(Concatenate(axis=3)([out, out3]))
        out = self.de5(Concatenate(axis=3)([out, out2]))
        out = self.de6(Concatenate(axis=3)([out, out1]))

        out += x
        return out


# Multi-Scale Feature Extraction (MSFE) - 5
class MSFE5(tf.Module):
    def __init__(self, mid_ch, out_ch, name=None):
        super(MSFE5, self).__init__(name=name)
        self.input_layer = Inconv(out_ch)

        # encoder
        self.en1 = CONV(mid_ch)
        self.en2 = CONV(mid_ch)
        self.en3 = CONV(mid_ch)
        self.en4 = CONV(mid_ch)
        self.en5 = CONV(mid_ch)

        # bottleneck
        self.ddense = dilated_dense_block(mid_ch, mid_ch, 6)

        # decoder
        self.de1 = Spconv(mid_ch)
        self.de2 = Spconv(mid_ch)
        self.de3 = Spconv(mid_ch)
        self.de4 = Spconv(mid_ch)
        self.de5 = Spconv(out_ch)

    def __call__(self, x):
        x = self.input_layer(x)

        # encoder
        out1 = self.en1(x)
        out2 = self.en2(out1)
        out3 = self.en3(out2)
        out4 = self.en4(out3)
        out5 = self.en5(out4)

        # bottleneck
        out = self.ddense(out5)

        # decoder
        out = self.de1(Concatenate(axis=3)([out, out5]))
        out = self.de2(Concatenate(axis=3)([out, out4]))
        out = self.de3(Concatenate(axis=3)([out, out3]))
        out = self.de4(Concatenate(axis=3)([out, out2]))
        out = self.de5(Concatenate(axis=3)([out, out1]))

        out += x
        return out


# Multi-Scale Feature Extraction (MSFE) - 4
class MSFE4(tf.Module):
    def __init__(self, mid_ch, out_ch, name=None):
        super(MSFE4, self).__init__(name=name)
        self.input_layer = Inconv(out_ch)

        # encoder
        self.en1 = CONV(mid_ch)
        self.en2 = CONV(mid_ch)
        self.en3 = CONV(mid_ch)
        self.en4 = CONV(mid_ch)

        # bottleneck
        self.ddense = dilated_dense_block(mid_ch, mid_ch, 6)

        # decoder
        self.de1 = Spconv(mid_ch)
        self.de2 = Spconv(mid_ch)
        self.de3 = Spconv(mid_ch)
        self.de4 = Spconv(out_ch)

    def __call__(self, x):
        x = self.input_layer(x)

        # encoder
        out1 = self.en1(x)
        out2 = self.en2(out1)
        out3 = self.en3(out2)
        out4 = self.en4(out3)

        # bottleneck
        out = self.ddense(out4)

        # decoder
        out = self.de1(Concatenate(axis=3)([out, out4]))
        out = self.de2(Concatenate(axis=3)([out, out3]))
        out = self.de3(Concatenate(axis=3)([out, out2]))
        out = self.de4(Concatenate(axis=3)([out, out1]))

        out += x
        return out


# Multi-Scale Feature Extraction (MSFE) - 3
class MSFE3(tf.Module):
    def __init__(self, mid_ch, out_ch, name=None):
        super(MSFE3, self).__init__(name=name)
        self.input_layer = Inconv(out_ch)

        # encoder
        self.en1 = CONV(mid_ch)
        self.en2 = CONV(mid_ch)
        self.en3 = CONV(mid_ch)

        # bottleneck
        self.ddense = dilated_dense_block(mid_ch, mid_ch, 6)

        # decoder
        self.de1 = Spconv(mid_ch)
        self.de2 = Spconv(mid_ch)
        self.de3 = Spconv(out_ch)

    def __call__(self, x):
        x = self.input_layer(x)

        # encoder
        out1 = self.en1(x)
        out2 = self.en2(out1)
        out3 = self.en3(out2)

        # bottleneck
        out = self.ddense(out3)

        # decoder
        out = self.de1(Concatenate(axis=3)([out, out3]))
        out = self.de2(Concatenate(axis=3)([out, out2]))
        out = self.de3(Concatenate(axis=3)([out, out1]))

        out += x
        return out


###################################################################################################################3

class NUNet(keras.Model):
    def __init__(self, in_ch=1, mid_ch=32, out_ch=64):
        super(NUNet, self).__init__()
        self.win_len = 512
        self.fft_len = 512
        self.stride = 128
        self.input_layer = Inconv(out_ch, name="Inconv")

        self.encoder_1 = keras.Sequential(
            [
                MSFE6(mid_ch=mid_ch, out_ch=out_ch, name="MSFE6"),
                DownSampling(out_ch, name="DownSampling")
            ],
            name='encoder_1'
        )

        self.encoder_2 = keras.Sequential(
            [
                MSFE5(mid_ch=mid_ch, out_ch=out_ch, name="MSFE5"),
                DownSampling(out_ch, name="DownSampling")
            ],
            name='encoder_2'
        )

        self.encoder_3 = keras.Sequential(
            [
                MSFE4(mid_ch=mid_ch, out_ch=out_ch, name="MSFE4"),
                DownSampling(out_ch, name="DownSampling")
            ],
            name='encoder_3'
        )

        self.encoder_4 = keras.Sequential(
            [
                MSFE4(mid_ch=mid_ch, out_ch=out_ch, name="MSFE4"),
                DownSampling(out_ch, name="DownSampling")
            ],
            name='encoder_4'
        )

        self.encoder_5 = keras.Sequential(
            [
                MSFE4(mid_ch=mid_ch, out_ch=out_ch, name="MSFE4"),
                DownSampling(out_ch, name="DownSampling")
            ],
            name='encoder_5'
        )

        self.encoder_6 = keras.Sequential(
            [
                MSFE3(mid_ch=mid_ch, out_ch=out_ch, name="MSFE3"),
                DownSampling(out_ch, name="DownSampling")
            ],
            name='encoder_6'
        )

        # Bottleneck block
        self.dense = keras.Sequential(
            [
                dilated_dense_block(out_ch, out_ch, 6, name="DDense")
            ],
            name='dense'
        )

        self.decoder_1 = keras.Sequential(
            [
                UpSampling(out_ch * 2, name="UpSampling"),
                MSFE3(mid_ch=mid_ch, out_ch=out_ch, name="MSFE3")
            ],
            name='decoder_1'
        )

        self.decoder_2 = keras.Sequential(
            [
                UpSampling(out_ch * 2, name="UpSampling"),
                MSFE4(mid_ch=mid_ch, out_ch=out_ch, name="MSFE4")
            ],
            name='decoder_2'
        )

        self.decoder_3 = keras.Sequential(
            [
                UpSampling(out_ch * 2, name="UpSampling"),
                MSFE4(mid_ch=mid_ch, out_ch=out_ch, name="MSFE4")
            ],
            name='decoder_3'
        )

        self.decoder_4 = keras.Sequential(
            [
                UpSampling(out_ch * 2, name="UpSampling"),
                MSFE4(mid_ch=mid_ch, out_ch=out_ch, name="MSFE4")
            ],
            name='decoder_4'
        )

        self.decoder_5 = keras.Sequential(
            [
                UpSampling(out_ch * 2, name="UpSampling"),
                MSFE5(mid_ch=mid_ch, out_ch=out_ch, name="MSFE5")
            ],
            name='decoder_5'
        )

        self.decoder_6 = keras.Sequential(
            [
                UpSampling(out_ch * 2, name="UpSampling"),
                MSFE6(mid_ch=mid_ch, out_ch=out_ch, name="MSFE6")
            ],
            name='decoder_6'
        )

        self.output_layer = Conv2D(in_ch, kernel_size=1)
        # self.hann_window = np.hanning(self.win_len).astype(np.float32)
        # self.hann_window[0], self.hann_window[511] = 1e-7, 1e-7
        # self.hann_window = tf.cast(self.hann_window, dtype=tf.float32)
        # self.hann_window = tf.constant(self.hann_window, dtype=tf.float32)  # [512,] --> (1, 1, 512)
        # self.DENORM = tf.constant((self.hann_window ** 2) * 4)
        # self.DENORM = (self.hann_window ** 2) * 4
        # self.hann_window = tf.reshape(self.hann_window, shape=[1, 1, -1])
        # self.DENORM = tf.reshape(self.DENORM, shape=[1, 1, -1])

    #
    # def unpreprocess_waveform(self, waveform, params):
    #     return tf.identity(waveform) * params[0]  # * 32768.0

    def call(self, x):
        # frames = tf.signal.frame(x, self.win_len, self.stride, pad_end=True)
        # frames = frames * self.hann_window
        frames = tf.signal.stft(x, frame_length=self.win_len, frame_step=self.stride, fft_length=self.fft_len,
                                window_fn=tf.signal.hann_window)
        # mags, phase, trans_mags = Lambda(stft_layer, name='stft')(frames)
        mags = tf.abs(frames)  # [None, None, 257]
        phase = tf.math.angle(frames)  # [None, None, 257]

        trans_mags = tf.expand_dims(mags, axis=3)
        trans_mags = trans_mags[:, :, 1:, :]

        # Input Layer
        hx = self.input_layer(trans_mags)  # [1, 372, 256, 64]

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
        out = self.dense(en6)  # [1, 372, 4, 64]

        # Decoder 1
        out = self.decoder_1(Concatenate(axis=3)([out, en6]))  # [1, 372, 8, 64]

        # Decoder 2
        out = self.decoder_2(Concatenate(axis=3)([out, en5]))  # [1, 372, 16, 64]

        # Decoder 3
        out = self.decoder_3(Concatenate(axis=3)([out, en4]))  # [1, 372, 32, 64]

        # Decoder 4
        out = self.decoder_4(Concatenate(axis=3)([out, en3]))

        # Decoder 5
        out = self.decoder_5(Concatenate(axis=3)([out, en2]))  # [1, 372, 128, 64]

        # Decoder 6
        out = self.decoder_6(Concatenate(axis=3)([out, en1]))  # [1, 372, 256, 64]

        # Output layer
        out = self.output_layer(out)  # [1, 372, 256, 1]

        est_mags = tf_masking(out, mags)  # T-F Masking
        # est_mags = spectral_mapping(out)  # spectral mapping

        # ISTFT
        recons = Lambda(ifft_layer, name='istft')([est_mags, phase])  # [Batch, n_frames, frame_size]
        # frames = frames * self.hann_window
        # frames = frames / self.DENORM  #
        # frames = tf.divide(frames, self.DENORM)
        # frames = tf.math.divide_no_nan(frames, self.DENORM)
        # recons = Lambda(overlap_add_layer, name='Overlap-add')(frames)
        # y = self.unpreprocess_waveform(recons, wparam)
        y = tf.clip_by_value(recons, -1, 1)
        # y = recons[:, :x.shape[1]]
        y = tf.squeeze(y)

        return y
