use super::tnn_types::TinyNNFP16;
use ndarray::{array, Array2};

pub fn get_mnist_image() -> Array2<TinyNNFP16> {
    array![
        [
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
        ],
        [
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
        ],
        [
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
        ],
        [
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.06270000338554382),
            TinyNNFP16::from_f32(0.5098000168800354),
            TinyNNFP16::from_f32(0.6607999801635742),
            TinyNNFP16::from_f32(0.6626999974250793),
            TinyNNFP16::from_f32(0.6607999801635742),
            TinyNNFP16::from_f32(0.24709999561309814),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
        ],
        [
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.24609999358654022),
            TinyNNFP16::from_f32(0.9157000184059143),
            TinyNNFP16::from_f32(0.7911999821662903),
            TinyNNFP16::from_f32(0.46860000491142273),
            TinyNNFP16::from_f32(0.3626999855041504),
            TinyNNFP16::from_f32(0.8813999891281128),
            TinyNNFP16::from_f32(0.8666999936103821),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
        ],
        [
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.009800000116229057),
            TinyNNFP16::from_f32(0.4431000053882599),
            TinyNNFP16::from_f32(0.9569000005722046),
            TinyNNFP16::from_f32(0.3666999936103821),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.5108000040054321),
            TinyNNFP16::from_f32(0.9901999831199646),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
        ],
        [
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.3314000070095062),
            TinyNNFP16::from_f32(0.9901999831199646),
            TinyNNFP16::from_f32(0.7088000178337097),
            TinyNNFP16::from_f32(0.04610000178217888),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.7519999742507935),
            TinyNNFP16::from_f32(0.6499999761581421),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
        ],
        [
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.3303999900817871),
            TinyNNFP16::from_f32(0.9656999707221985),
            TinyNNFP16::from_f32(0.7930999994277954),
            TinyNNFP16::from_f32(0.08529999852180481),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.11180000007152557),
            TinyNNFP16::from_f32(0.907800018787384),
            TinyNNFP16::from_f32(0.14219999313354492),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
        ],
        [
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.8980000019073486),
            TinyNNFP16::from_f32(0.9265000224113464),
            TinyNNFP16::from_f32(0.9656999707221985),
            TinyNNFP16::from_f32(0.13920000195503235),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.006899999920278788),
            TinyNNFP16::from_f32(0.6901999711990356),
            TinyNNFP16::from_f32(0.5098000168800354),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
        ],
        [
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.6843000054359436),
            TinyNNFP16::from_f32(0.8048999905586243),
            TinyNNFP16::from_f32(0.40290001034736633),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.5146999955177307),
            TinyNNFP16::from_f32(0.5755000114440918),
            TinyNNFP16::from_f32(0.05490000173449516),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
        ],
        [
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.4293999969959259),
            TinyNNFP16::from_f32(0.8264999985694885),
            TinyNNFP16::from_f32(0.9814000129699707),
            TinyNNFP16::from_f32(0.9901999831199646),
            TinyNNFP16::from_f32(0.9930999875068665),
            TinyNNFP16::from_f32(0.7206000089645386),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
        ],
        [
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.21080000698566437),
            TinyNNFP16::from_f32(0.3824000060558319),
            TinyNNFP16::from_f32(0.33730000257492065),
            TinyNNFP16::from_f32(0.011800000444054604),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
        ],
        [
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
        ],
        [
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
            TinyNNFP16::from_f32(0.0),
        ],
    ]
}

pub fn get_mnist_convolve_0_params() -> Array2<TinyNNFP16> {
    array![
        [
            TinyNNFP16::from_f32(-0.008700000122189522),
            TinyNNFP16::from_f32(0.00139999995008111),
        ],
        [
            TinyNNFP16::from_f32(-0.09920000284910202),
            TinyNNFP16::from_f32(-0.3628000020980835),
        ],
        [
            TinyNNFP16::from_f32(0.028200000524520874),
            TinyNNFP16::from_f32(0.049300000071525574),
        ],
        [
            TinyNNFP16::from_f32(-0.0763000026345253),
            TinyNNFP16::from_f32(0.43709999322891235),
        ]
    ]
}

pub fn get_mnist_convolve_1_params() -> Array2<TinyNNFP16> {
    array![
        [
          TinyNNFP16::from_f32(-0.462799996137619),
          TinyNNFP16::from_f32(0.7610999941825867),
        ],
        [
          TinyNNFP16::from_f32(0.27239999175071716),
          TinyNNFP16::from_f32(0.6888999938964844),
        ],
        [
          TinyNNFP16::from_f32(0.8213000297546387),
          TinyNNFP16::from_f32(0.09920000284910202),
        ],
        [
          TinyNNFP16::from_f32(0.07729999721050262),
          TinyNNFP16::from_f32(-0.4505999982357025),
        ]
    ]
}

pub fn get_incrementing_test_image() -> Array2<TinyNNFP16> {
    array![
        [
            TinyNNFP16::from_f32(0.000000),
            TinyNNFP16::from_f32(8.000000),
            TinyNNFP16::from_f32(16.000000),
            TinyNNFP16::from_f32(24.000000),
            TinyNNFP16::from_f32(32.000000),
            TinyNNFP16::from_f32(40.000000),
            TinyNNFP16::from_f32(48.000000),
            TinyNNFP16::from_f32(56.000000),
        ],
        [
            TinyNNFP16::from_f32(1.000000),
            TinyNNFP16::from_f32(9.000000),
            TinyNNFP16::from_f32(17.000000),
            TinyNNFP16::from_f32(25.000000),
            TinyNNFP16::from_f32(33.000000),
            TinyNNFP16::from_f32(41.000000),
            TinyNNFP16::from_f32(49.000000),
            TinyNNFP16::from_f32(57.000000),
        ],
        [
            TinyNNFP16::from_f32(2.000000),
            TinyNNFP16::from_f32(10.000000),
            TinyNNFP16::from_f32(18.000000),
            TinyNNFP16::from_f32(26.000000),
            TinyNNFP16::from_f32(34.000000),
            TinyNNFP16::from_f32(42.000000),
            TinyNNFP16::from_f32(50.000000),
            TinyNNFP16::from_f32(58.000000),
        ],
        [
            TinyNNFP16::from_f32(3.000000),
            TinyNNFP16::from_f32(11.000000),
            TinyNNFP16::from_f32(19.000000),
            TinyNNFP16::from_f32(27.000000),
            TinyNNFP16::from_f32(35.000000),
            TinyNNFP16::from_f32(43.000000),
            TinyNNFP16::from_f32(51.000000),
            TinyNNFP16::from_f32(59.000000),
        ],
        [
            TinyNNFP16::from_f32(4.000000),
            TinyNNFP16::from_f32(12.000000),
            TinyNNFP16::from_f32(20.000000),
            TinyNNFP16::from_f32(28.000000),
            TinyNNFP16::from_f32(36.000000),
            TinyNNFP16::from_f32(44.000000),
            TinyNNFP16::from_f32(52.000000),
            TinyNNFP16::from_f32(60.000000),
        ],
        [
            TinyNNFP16::from_f32(5.000000),
            TinyNNFP16::from_f32(13.000000),
            TinyNNFP16::from_f32(21.000000),
            TinyNNFP16::from_f32(29.000000),
            TinyNNFP16::from_f32(37.000000),
            TinyNNFP16::from_f32(45.000000),
            TinyNNFP16::from_f32(53.000000),
            TinyNNFP16::from_f32(61.000000),
        ],
        [
            TinyNNFP16::from_f32(6.000000),
            TinyNNFP16::from_f32(14.000000),
            TinyNNFP16::from_f32(22.000000),
            TinyNNFP16::from_f32(30.000000),
            TinyNNFP16::from_f32(38.000000),
            TinyNNFP16::from_f32(46.000000),
            TinyNNFP16::from_f32(54.000000),
            TinyNNFP16::from_f32(62.000000),
        ],
        [
            TinyNNFP16::from_f32(7.000000),
            TinyNNFP16::from_f32(15.000000),
            TinyNNFP16::from_f32(23.000000),
            TinyNNFP16::from_f32(31.000000),
            TinyNNFP16::from_f32(39.000000),
            TinyNNFP16::from_f32(47.000000),
            TinyNNFP16::from_f32(55.000000),
            TinyNNFP16::from_f32(63.000000),
        ],
    ]
}

pub fn get_half_const_convolve_params() -> Array2<TinyNNFP16> {
    array![
        [TinyNNFP16::from_f32(0.5), TinyNNFP16::from_f32(0.5),],
        [TinyNNFP16::from_f32(0.5), TinyNNFP16::from_f32(0.5),],
        [TinyNNFP16::from_f32(0.5), TinyNNFP16::from_f32(0.5),],
        [TinyNNFP16::from_f32(0.5), TinyNNFP16::from_f32(0.5),]
    ]
}
