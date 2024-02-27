import math
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import glob
import time
from datetime import datetime
import tensorflow as tf
import numpy as np
import configargparse
from tensorflow.keras import layers
from tensorflow.keras import datasets
import tensorflow_compression as tfc


def psnr_metric(x_in, x_out):
    if type(x_in) is list:
        img_in = x_in[0]
    else:
        img_in = x_in
    return tf.image.psnr(img_in, x_out, max_val=1.0)


class Encoder(layers.Layer):
    """Build encoder arch"""

    def __init__(self, conv_depth, name="encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.data_format = "channels_last"
        self.sublayers = [
            tfc.SignalConv2D(
                16,
                (5, 5),
                name="conv_1",
                corr=True,
                strides_down=2,
                padding="same_zeros",
                use_bias=True,
            ),
            layers.PReLU(shared_axes=[1, 2]),
            tfc.SignalConv2D(
                32,
                (5, 5),
                name="conv_2",
                corr=True,
                strides_down=2,
                padding="same_zeros",
                use_bias=True,
            ),
            layers.PReLU(shared_axes=[1, 2]),
            tfc.SignalConv2D(
                32,
                (5, 5),
                name="conv_3",
                corr=True,
                strides_down=1,
                padding="same_zeros",
                use_bias=True,
            ),
            layers.PReLU(shared_axes=[1, 2]),
            tfc.SignalConv2D(
                32,
                (5, 5),
                name="conv_4",
                corr=True,
                strides_down=1,
                padding="same_zeros",
                use_bias=True,
            ),
            layers.PReLU(shared_axes=[1, 2]),
            tfc.SignalConv2D(
                conv_depth,
                (5, 5),
                name="conv_5",
                corr=True,
                strides_down=1,
                padding="same_zeros",
                use_bias=True,
                activation=None,
            ),
        ]

    def call(self, x):
        for sublayer in self.sublayers:
            x = sublayer(x)
        return x


class Decoder(layers.Layer):
    """Build decoder arch"""

    def __init__(self, n_channels, name="decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.data_format = "channels_last"
        self.sublayers = [
            tfc.SignalConv2D(
                32,
                (5, 5),
                name="conv_1",
                corr=False,
                strides_up=1,
                padding="same_zeros",
                use_bias=True,
            ),
            layers.PReLU(shared_axes=[1, 2]),
            tfc.SignalConv2D(
                32,
                (5, 5),
                name="conv_2",
                corr=False,
                strides_up=1,
                padding="same_zeros",
                use_bias=True,
            ),
            layers.PReLU(shared_axes=[1, 2]),
            tfc.SignalConv2D(
                32,
                (5, 5),
                name="conv_3",
                corr=False,
                strides_up=1,
                padding="same_zeros",
                use_bias=True,
            ),
            layers.PReLU(shared_axes=[1, 2]),
            tfc.SignalConv2D(
                16,
                (5, 5),
                name="conv_4",
                corr=False,
                strides_up=2,
                padding="same_zeros",
                use_bias=True,
            ),
            layers.PReLU(shared_axes=[1, 2]),
            tfc.SignalConv2D(
                n_channels,
                (5, 5),
                name="conv_5",
                corr=False,
                strides_up=2,
                padding="same_zeros",
                use_bias=True,
                activation=tf.nn.sigmoid,
            ),
        ]

    def call(self, x):
        for sublayer in self.sublayers:
            x = sublayer(x)
        return x


def max_Rate(k, n, snr):
    """Implements the maximum rate R (banwidth of the channel).
    Args:
        k: channel bandwidth
        n: image dimension (source bandwidth)
        snr: channel signal-to-noise rate 
    Returns:
        Rmax: Max bit rate 
    """
    Rmax = np.divide(k,n) * math.log2(1+(10**(snr/10)))

    return Rmax


def real_awgn(x, stddev):
    """Implements the real additive white gaussian noise channel.
    Args:
        x: channel input symbols
        stddev: standard deviation of noise
    Returns:
        y: noisy channel output symbols
    """
    # additive white gaussian noise
    awgn = tf.random.normal(tf.shape(x), 0, stddev, dtype=tf.float32)
    y = x + awgn

    return y


def fading(x, stddev, h=None):
    """Implements the fading channel with multiplicative fading and
    additive white gaussian noise.
    Args:
        x: channel input symbols
        stddev: standard deviation of noise
    Returns:
        y: noisy channel output symbols
    """
    # channel gain
    if h is None:
        h = tf.complex(
            tf.random.normal([tf.shape(x)[0], 1], 0, 1 / np.sqrt(2)),
            tf.random.normal([tf.shape(x)[0], 1], 0, 1 / np.sqrt(2)),
        )

    # additive white gaussian noise
    awgn = tf.complex(
        tf.random.normal(tf.shape(x), 0, 1 / np.sqrt(2)),
        tf.random.normal(tf.shape(x), 0, 1 / np.sqrt(2)),
    )

    return (h * x + stddev * awgn), h


class Channel(layers.Layer):
    def __init__(self, channel_type, channel_snr, name="channel", **kwargs):
        super(Channel, self).__init__(name=name, **kwargs)
        self.channel_type = channel_type
        self.channel_snr = channel_snr

    def call(self, inputs):
        (encoded_img, prev_h) = inputs
        inter_shape = tf.shape(encoded_img)
        # reshape array to [-1, dim_z]
        z = layers.Flatten()(encoded_img)
        # convert from snr to std
        print("channel_snr: {}".format(self.channel_snr))
        noise_stddev = np.sqrt(10 ** (-self.channel_snr / 10))

        # Add channel noise
        if self.channel_type == "awgn":
            dim_z = tf.shape(z)[1]
            # normalize latent vector so that the average power is 1
            z_in = tf.sqrt(tf.cast(dim_z, dtype=tf.float32)) * tf.nn.l2_normalize(
                z, axis=1)
            z_out = real_awgn(z_in, noise_stddev)
            h = tf.ones_like(z_in)  # h just makes sense on fading channels

        elif self.channel_type == "fading":
            dim_z = tf.shape(z)[1] // 2
            # convert z to complex representation
            z_in = tf.complex(z[:, :dim_z], z[:, dim_z:])
            # normalize the latent vector so that the average power is 1
            z_norm = tf.reduce_sum(
                tf.math.real(z_in * tf.math.conj(z_in)), axis=1, keepdims=True
            )
            z_in = z_in * tf.complex(
                tf.sqrt(tf.cast(dim_z, dtype=tf.float32) / z_norm), 0.0
            )
            z_out, h = fading(z_in, noise_stddev, prev_h)
            # convert back to real
            z_out = tf.concat([tf.math.real(z_out), tf.math.imag(z_out)], 1)

        # convert signal back to intermediate shape
        z_out = tf.reshape(z_out, inter_shape)

        return z_out, h

class D_JSCC(layers.Layer):
    """Build D-JSCC arch"""
    def __init__(
        self,
        channel_snr,
        conv_depth,
        channel_type,
        name="deep_jscc",
        **kwargs
    ):
        super(D_JSCC, self).__init__(name=name, **kwargs)

        n_channels = 3  # For RGB, change this if working with BW images
        self.encoder = Encoder(conv_depth)
        self.decoder = Decoder(n_channels, name="decoder_output")
        self.channel = Channel(channel_type, channel_snr, name="channel_output")

    def call(self, inputs):
        
        # inputs is just the original image
        img_in = img = inputs
        prev_chn_gain = None

        chn_in = self.encoder(img_in)
        chn_out, chn_gain = self.channel((chn_in, prev_chn_gain))

        decoded_img = self.decoder(chn_out)

        # keep track of some metrics
        self.add_metric(
            tf.image.psnr(img, decoded_img, max_val=1.0),
            aggregation="mean",
            name="psnr",
        )

        self.add_metric(
            tf.reduce_mean(tf.math.square(img - decoded_img)),
            aggregation="mean",
            name="mse",
        )
    
        return (decoded_img, chn_out, chn_gain)

    def change_channel_snr(self, channel_snr):
        self.channel.channel_snr = channel_snr

    def change_feedback_snr(self, feedback_snr):
        self.feedback_snr = feedback_snr


def main(args):
    
    # get train and test CIFAR dataset
    x_train, x_test = get_dataset(args.number_of_train_image,args.number_of_test_image)
    
    if args.delete_previous_model and tf.io.gfile.exists(args.model_dir):
        print("Deleting previous model files at {}".format(args.model_dir))
        tf.io.gfile.rmtree(args.model_dir)
        tf.io.gfile.makedirs(args.model_dir)
    else:
        print("Starting new model at {}".format(args.model_dir))
        tf.io.gfile.makedirs(args.model_dir)

    # load model
    prev_layer_out = None
    # add input placeholder to please keras
    img = tf.keras.Input(shape=(None, None, 3))

    channel_snr = args.channel_snr_train

    # Max R (bit rate/bandwidth) of the AWGN Channel given CIFAR dataset
    image_dim = 32 * 32 * 3
    channel_Rmax = max_Rate(args.conv_depth, image_dim, channel_snr) 
    
    # checkpoint
    ckpt_file = os.path.join(args.model_dir, "ckpt")
    
    # D-JSCC model object
    ae_layer = D_JSCC(
        channel_snr,
        int(args.conv_depth),
        args.channel,
    )

    layer_output = ae_layer(img)
    
    (
        decoded_img,
        _chn_out,
        _chn_gain,
    ) = layer_output

    model = tf.keras.Model(inputs=img, outputs=decoded_img)
    
    model_metrics = [
        tf.keras.metrics.MeanSquaredError(),
        psnr_metric,
    ]
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learn_rate),
        loss="mse",
        metrics=model_metrics,
    )
    
    print(model.summary())
    
    checkpoint_path = os.path.join(args.model_dir, "ckpt")
    checkpoint_dir = os.path.dirname(checkpoint_path)
    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,\
                                                     save_weights_only=True, verbose=1)
    model.load_weights("/media/bohnsix/D-JSCC/train_logs/checkpoint")

    model.fit(
        x_train,
        x_train,
        # epochs=args.train_epochs,
        epochs=1000,
        callbacks=[cp_callback],
        verbose=2,
        batch_size=args.batch_size_train,
    )

    model.trainable = False


    print("<----------EVALUATION--------->")
    # eval the model
    out_eval = model.evaluate(x_test,x_test, verbose=2,batch_size=args.batch_size_test)
    for m, v in zip(model.metrics_names, out_eval):
        met_name = "_".join(["eval", m])
        print("{}={}".format(met_name, v), end=" ")
    print("\n")
    

def get_dataset(no_of_train_images,no_of_test_images):
    
    # load train and test images of CIFAR-10 dataset
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    train_labels, test_labels = train_labels, test_labels
    # Normalize pixel values to be between 0 and 1 for all the images
    x_train, x_tst = train_images[:no_of_train_images] / 255.0, test_images[:no_of_test_images] / 255.0

    return x_train, x_tst


if __name__ == "__main__":
    # parse args
    p = configargparse.ArgParser()
    p.add(
        "-c",
        "--my-config",
        required=False,
        is_config_file=True,
        help="config file path",
    )
    p.add(
        "--conv_depth",
        type=float,
        default=8,
        help=(
            "Number of channels of last conv layer, used to define the "
            "compression rate: k/n=c_out/(16*3)"
        ),

    )
    p.add(
        "--channel",
        type=str,
        default="fading",
        choices=["awgn", "fading"],
        help="Model of channel used (awgn, fading)",
    )
    p.add(
        "--model_dir",
        type=str,
        default="./train_logs",
        help=("The location of the model checkpoint files."),
    )
    p.add(
        "--delete_previous_model",
        action="store_true",
        default=False,
        help=("If model_dir has checkpoints, delete it before" "starting new run"),
    )
    p.add(
        "--channel_snr_train",
        type=float,
        default=10,
        help="target SNR of channel during training (dB)",
    )
    p.add(
        "--number_of_train_image",
        type=int,
        default=5000,
        help="Number of training images during training ",
    )
    p.add(
        "--number_of_test_image",
        type=int,
        default=1000,
        help="Number of test images during testing ",
    )
    p.add(
        "--learn_rate",
        type=float,
        default=0.001,
        help="Learning rate for Adam optimizer",
    )
    p.add(
        "--train_epochs",
        type=int,
        default=2500,
        help=(
            "The number of epochs used to train (each epoch goes over the whole dataset)"
        ),
    )
    p.add("--batch_size_train", type=int, default=64, help="Batch size for training")
    p.add("--batch_size_test", type=int, default=64, help="Batch size for testing")

    args = p.parse_args()

    print("##############D-JSCC#########################")
    for arg, value in sorted(vars(args).items()):
        print("{}: {}".format(arg, value))
    print("#############################################")
    main(args)



"""
conda activate deepjscc_bohnsix

python -u main.py --channel awgn --batch_size_train 512 --batch_size_test 512 > log2000.log
"""
