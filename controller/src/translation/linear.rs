//! Linear layer translation.
//!
//! Translates fully-connected (linear) layers to TNN mul_acc operations.

use crate::cnn::schema::LinearLayer;
use crate::error::ControllerError;
use crate::fp16::TinyNNFP16;
use crate::tnn_ops::mul_acc_stream;

/// Translates a linear layer to TNN mul_acc operations.
/// For each output neuron, generates one mul_acc operation.
///
/// Input: Vector of in_features values
/// Output: Vector of out_features values
///
/// Each output\[i\] = relu(sum(input\[j\] * weight\[i,j\]) + bias\[i\])
pub fn translate_linear(
    layer: &LinearLayer,
    input: &[TinyNNFP16],
    weights: &[TinyNNFP16],
    bias: &[TinyNNFP16],
) -> Result<Vec<Vec<u16>>, ControllerError> {
    if input.len() != layer.in_features {
        return Err(ControllerError::ShapeMismatch(format!(
            "Linear layer {} expects {} inputs, got {}",
            layer.name, layer.in_features, input.len()
        )));
    }

    if weights.len() != layer.in_features * layer.out_features {
        return Err(ControllerError::ShapeMismatch(format!(
            "Linear layer {} expects {} weights, got {}",
            layer.name,
            layer.in_features * layer.out_features,
            weights.len()
        )));
    }

    if bias.len() != layer.out_features {
        return Err(ControllerError::ShapeMismatch(format!(
            "Linear layer {} expects {} bias values, got {}",
            layer.name, layer.out_features, bias.len()
        )));
    }

    let mut streams = Vec::with_capacity(layer.out_features);

    for out_idx in 0..layer.out_features {
        // Build interleaved (input, weight) pairs for this output neuron
        let mut pairs = Vec::with_capacity(layer.in_features * 2);

        // Weights are stored in row-major order: [out_features, in_features]
        // So weights for output out_idx are at indices [out_idx * in_features .. (out_idx+1) * in_features]
        let weight_start = out_idx * layer.in_features;

        for in_idx in 0..layer.in_features {
            pairs.push(input[in_idx]);
            pairs.push(weights[weight_start + in_idx]);
        }

        let stream = mul_acc_stream(&pairs, bias[out_idx], layer.relu);
        streams.push(stream);
    }

    Ok(streams)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_translate_linear() {
        let layer = LinearLayer {
            name: "test_fc".to_string(),
            in_features: 3,
            out_features: 2,
            relu: false,
            weights: None,
            bias: None,
        };

        let input = vec![
            TinyNNFP16::from_f32(1.0),
            TinyNNFP16::from_f32(2.0),
            TinyNNFP16::from_f32(3.0),
        ];

        // Weights: [out_features, in_features] = [2, 3]
        // out=0: [w00, w01, w02] = [0.1, 0.2, 0.3]
        // out=1: [w10, w11, w12] = [0.4, 0.5, 0.6]
        let weights = vec![
            TinyNNFP16::from_f32(0.1),
            TinyNNFP16::from_f32(0.2),
            TinyNNFP16::from_f32(0.3),
            TinyNNFP16::from_f32(0.4),
            TinyNNFP16::from_f32(0.5),
            TinyNNFP16::from_f32(0.6),
        ];

        let bias = vec![TinyNNFP16::from_f32(0.0), TinyNNFP16::from_f32(0.1)];

        let streams = translate_linear(&layer, &input, &weights, &bias).unwrap();

        assert_eq!(streams.len(), 2); // One stream per output neuron

        // Each stream should have: cmd + bias + 6 values (3 pairs) + NaN = 9 words
        assert_eq!(streams[0].len(), 9);
        assert_eq!(streams[1].len(), 9);
    }
}
