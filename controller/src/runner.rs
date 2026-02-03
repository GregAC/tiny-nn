//! CNN execution orchestration.
//!
//! This module provides [`CnnRunner`] for generating TNN command sequences
//! from CNN model descriptions, and [`plan_execution`] for analyzing models.
//!
//! ## Live Execution
//!
//! The runner can execute models in two modes:
//!
//! 1. **Offline generation** (`generate_commands`): Generates command streams
//!    for later execution, using placeholder values where actual TNN outputs
//!    would be needed.
//!
//! 2. **Live execution** (`run`): Executes the model layer by layer through
//!    a `TnnInterface`, collecting real outputs and feeding them to subsequent
//!    layers.

use std::collections::HashMap;
use std::path::Path;

use crate::cnn::{load_weights, CnnModel, DataSource, Layer, TensorShape};
use crate::comm::{recv_fp16_values_with_padding, CommandBuffer, TnnInterface};
use crate::error::ControllerError;
use crate::fp16::TinyNNFP16;
use crate::translation::{translate_avg_pool2d, translate_conv2d, translate_linear, translate_max_pool2d};

/// Results from live model execution.
///
/// Contains the final output and optionally intermediate layer outputs.
#[derive(Debug, Clone)]
pub struct LayerOutputs {
    /// Final model output values.
    pub final_output: Vec<TinyNNFP16>,
    /// Intermediate layer outputs, keyed by layer name.
    /// Only populated if `capture_intermediate` was true during execution.
    pub layer_outputs: HashMap<String, Vec<TinyNNFP16>>,
}

/// Orchestrates the execution of a CNN model on TNN.
///
/// The runner loads a model and its weights, then generates TNN command
/// streams for inference or executes live through a TNN interface.
///
/// # Example
///
/// ```no_run
/// use controller::{load_model, CnnRunner, TinyNNFP16};
///
/// let model = load_model("model.toml").unwrap();
/// let runner = CnnRunner::new(model, ".");
///
/// let input = vec![TinyNNFP16::from_f32(1.0); 64];
/// let commands = runner.generate_commands(&input).unwrap();
/// commands.write_to_file("commands.hex").unwrap();
/// ```
pub struct CnnRunner {
    model: CnnModel,
    base_path: std::path::PathBuf,
}

impl CnnRunner {
    pub fn new<P: AsRef<Path>>(model: CnnModel, base_path: P) -> Self {
        CnnRunner {
            model,
            base_path: base_path.as_ref().to_path_buf(),
        }
    }

    /// Get the model's input shape.
    pub fn input_shape(&self) -> TensorShape {
        TensorShape::from_chw(self.model.metadata.input_shape)
    }

    /// Generate all command streams for the model given an input.
    /// Returns all commands concatenated into a single buffer.
    pub fn generate_commands(
        &self,
        input: &[TinyNNFP16],
    ) -> Result<CommandBuffer, ControllerError> {
        let mut buffer = CommandBuffer::new();
        let mut current_shape = self.input_shape();

        if input.len() != current_shape.size() {
            return Err(ControllerError::ShapeMismatch(format!(
                "Model expects {} input values, got {}",
                current_shape.size(),
                input.len()
            )));
        }

        // Track current activations (in practice, these would come from TNN outputs)
        // For command generation, we just track shape.
        let mut _current_activations = input.to_vec();

        for layer in &self.model.layers {
            match layer {
                Layer::Conv2d(conv) => {
                    let weights = self.load_layer_weights(&conv.weights, &conv.name)?;
                    let bias = self.load_layer_bias(&conv.bias, &conv.name)?;

                    let translation = translate_conv2d(
                        conv,
                        &current_shape,
                        &_current_activations,
                        &weights,
                        &bias,
                    )?;

                    // Add all convolve streams
                    for stream in &translation.convolve_streams {
                        buffer.extend(stream);
                    }

                    // Add accumulate streams
                    for stream in &translation.accumulate_streams {
                        buffer.extend(stream);
                    }

                    current_shape = current_shape.after_conv2d(conv.out_channels, conv.kernel_size);
                }

                Layer::Linear(linear) => {
                    let weights = self.load_layer_weights(&linear.weights, &linear.name)?;
                    let bias = self.load_layer_bias(&linear.bias, &linear.name)?;

                    let input_size = current_shape
                        .as_vector()
                        .ok_or_else(|| {
                            ControllerError::ShapeMismatch(
                                "Linear layer expects flattened input".into(),
                            )
                        })?;

                    // Create placeholder input for command generation
                    let placeholder_input: Vec<TinyNNFP16> =
                        vec![TinyNNFP16::zero(); input_size];

                    let streams =
                        translate_linear(linear, &placeholder_input, &weights, &bias)?;

                    for stream in &streams {
                        buffer.extend(stream);
                    }

                    current_shape = TensorShape::Vector {
                        size: linear.out_features,
                    };
                }

                Layer::MaxPool2d(pool) => {
                    // Create placeholder input for command generation
                    let placeholder_input: Vec<TinyNNFP16> =
                        vec![TinyNNFP16::zero(); current_shape.size()];

                    let stream =
                        translate_max_pool2d(pool, &current_shape, &placeholder_input)?;

                    buffer.extend(&stream);

                    current_shape = current_shape.after_pool(pool.kernel_size, pool.stride);
                }

                Layer::AvgPool2d(pool) => {
                    // Create placeholder input for command generation
                    let placeholder_input: Vec<TinyNNFP16> =
                        vec![TinyNNFP16::zero(); current_shape.size()];

                    let stream =
                        translate_avg_pool2d(pool, &current_shape, &placeholder_input)?;

                    buffer.extend(&stream);

                    current_shape = current_shape.after_pool(pool.kernel_size, pool.stride);
                }

                Layer::Flatten(_) => {
                    // No TNN operation, just update shape
                    current_shape = current_shape.after_flatten();
                }
            }
        }

        Ok(buffer)
    }

    /// Load weights from a data source, or return zeros if no source specified.
    fn load_layer_weights(
        &self,
        source: &Option<DataSource>,
        layer_name: &str,
    ) -> Result<Vec<TinyNNFP16>, ControllerError> {
        match source {
            Some(src) => load_weights(src, &self.base_path),
            None => Err(ControllerError::MissingWeights(layer_name.to_string())),
        }
    }

    /// Load bias from a data source, or return zeros if no source specified.
    fn load_layer_bias(
        &self,
        source: &Option<DataSource>,
        layer_name: &str,
    ) -> Result<Vec<TinyNNFP16>, ControllerError> {
        match source {
            Some(src) => load_weights(src, &self.base_path),
            None => Err(ControllerError::MissingWeights(format!(
                "{} (bias)",
                layer_name
            ))),
        }
    }

    /// Execute the model through a TNN interface.
    ///
    /// This performs live execution, sending commands layer by layer and
    /// collecting the actual TNN outputs to feed to subsequent layers.
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor values (must match model's input shape)
    /// * `interface` - TNN interface for communication
    /// * `capture_intermediate` - Whether to capture outputs from each layer
    ///
    /// # Returns
    ///
    /// [`LayerOutputs`] containing the final output and optionally intermediate
    /// layer outputs.
    pub fn run<T: TnnInterface>(
        &self,
        input: &[TinyNNFP16],
        interface: &mut T,
        capture_intermediate: bool,
    ) -> Result<LayerOutputs, ControllerError> {
        let mut current_shape = self.input_shape();
        let mut layer_outputs = HashMap::new();

        if input.len() != current_shape.size() {
            return Err(ControllerError::ShapeMismatch(format!(
                "Model expects {} input values, got {}",
                current_shape.size(),
                input.len()
            )));
        }

        let mut current_activations = input.to_vec();

        for layer in &self.model.layers {
            let layer_name = layer.name().to_string();
            let out_shape: TensorShape;

            match layer {
                Layer::Conv2d(conv) => {
                    let weights = self.load_layer_weights(&conv.weights, &conv.name)?;
                    let bias = self.load_layer_bias(&conv.bias, &conv.name)?;

                    let translation = translate_conv2d(
                        conv,
                        &current_shape,
                        &current_activations,
                        &weights,
                        &bias,
                    )?;

                    out_shape = current_shape.after_conv2d(conv.out_channels, conv.kernel_size);
                    let (_, out_h, out_w) = out_shape.as_image().unwrap();
                    let out_pixels = out_h * out_w;

                    // For conv2d, we execute all convolve operations and collect
                    // partial results, then accumulate per output channel
                    let kernel_size = conv.kernel_size;
                    let num_tiles_x = (kernel_size + 3) / 4;
                    let num_tiles_y = (kernel_size + 1) / 2;
                    let tiles_per_kernel = num_tiles_x * num_tiles_y;
                    let partials_per_pixel = conv.in_channels * tiles_per_kernel;

                    // Execute all convolve operations and collect partial results
                    let mut all_partials: Vec<Vec<TinyNNFP16>> = vec![Vec::new(); conv.out_channels];

                    let mut convolve_idx = 0;
                    for out_ch in 0..conv.out_channels {
                        for _in_ch in 0..conv.in_channels {
                            for _tile in 0..tiles_per_kernel {
                                // Send convolve command
                                interface.send_stream(&translation.convolve_streams[convolve_idx])?;

                                // Receive outputs (out_pixels values)
                                let results = recv_fp16_values_with_padding(interface, out_pixels)?;
                                all_partials[out_ch].extend(results);

                                convolve_idx += 1;
                            }
                        }
                    }

                    // Now execute accumulate for each output channel
                    let mut channel_outputs = Vec::new();
                    for out_ch in 0..conv.out_channels {
                        // Build accumulate stream with actual partial values
                        // The partials need to be reordered for accumulation:
                        // For each pixel, group all partials_per_pixel values together
                        let mut accum_values = Vec::new();
                        for pixel in 0..out_pixels {
                            for partial in 0..partials_per_pixel {
                                accum_values.push(all_partials[out_ch][partial * out_pixels + pixel]);
                            }
                        }

                        let accum_stream = crate::tnn_ops::accumulate_stream(
                            &accum_values,
                            partials_per_pixel,
                            bias[out_ch],
                            conv.relu,
                        );
                        interface.send_stream(&accum_stream)?;

                        // Receive accumulated results
                        let results = recv_fp16_values_with_padding(interface, out_pixels)?;
                        channel_outputs.extend(results);
                    }

                    current_activations = channel_outputs;
                }

                Layer::Linear(linear) => {
                    let weights = self.load_layer_weights(&linear.weights, &linear.name)?;
                    let bias = self.load_layer_bias(&linear.bias, &linear.name)?;

                    // Translate and execute each output neuron
                    let streams =
                        translate_linear(linear, &current_activations, &weights, &bias)?;

                    let mut outputs = Vec::with_capacity(linear.out_features);
                    for stream in &streams {
                        interface.send_stream(stream)?;
                        // Each mul_acc produces exactly 1 output
                        let result = recv_fp16_values_with_padding(interface, 1)?;
                        outputs.extend(result);
                    }

                    out_shape = TensorShape::Vector {
                        size: linear.out_features,
                    };
                    current_activations = outputs;
                }

                Layer::MaxPool2d(pool) => {
                    let stream =
                        translate_max_pool2d(pool, &current_shape, &current_activations)?;

                    interface.send_stream(&stream)?;

                    out_shape = current_shape.after_pool(pool.kernel_size, pool.stride);
                    let results = recv_fp16_values_with_padding(interface, out_shape.size())?;
                    current_activations = results;
                }

                Layer::AvgPool2d(pool) => {
                    let stream =
                        translate_avg_pool2d(pool, &current_shape, &current_activations)?;

                    interface.send_stream(&stream)?;

                    out_shape = current_shape.after_pool(pool.kernel_size, pool.stride);
                    let results = recv_fp16_values_with_padding(interface, out_shape.size())?;
                    current_activations = results;
                }

                Layer::Flatten(_) => {
                    // No TNN operation, just update shape
                    out_shape = current_shape.after_flatten();
                }
            }

            if capture_intermediate {
                layer_outputs.insert(layer_name, current_activations.clone());
            }

            current_shape = out_shape;
        }

        Ok(LayerOutputs {
            final_output: current_activations,
            layer_outputs,
        })
    }

    /// Get the model name.
    pub fn model_name(&self) -> &str {
        &self.model.metadata.name
    }
}

/// Describes the execution plan for a single layer.
///
/// Used by [`plan_execution`] to show what operations each layer requires.
#[derive(Debug)]
pub enum LayerPlan {
    Conv2d {
        name: String,
        in_shape: TensorShape,
        out_shape: TensorShape,
        num_convolve_ops: usize,
        num_accumulate_ops: usize,
    },
    Linear {
        name: String,
        in_size: usize,
        out_size: usize,
        num_mul_acc_ops: usize,
    },
    MaxPool2d {
        name: String,
        in_shape: TensorShape,
        out_shape: TensorShape,
    },
    AvgPool2d {
        name: String,
        in_shape: TensorShape,
        out_shape: TensorShape,
    },
    Flatten {
        name: String,
        in_shape: TensorShape,
        out_size: usize,
    },
}

/// Generate an execution plan for a model without generating actual commands.
///
/// Returns a list of [`LayerPlan`] entries describing what operations each
/// layer requires and how tensor shapes change through the network.
///
/// # Example
///
/// ```no_run
/// use controller::{load_model, plan_execution};
///
/// let model = load_model("model.toml").unwrap();
/// for plan in plan_execution(&model) {
///     println!("{:?}", plan);
/// }
/// ```
pub fn plan_execution(model: &CnnModel) -> Vec<LayerPlan> {
    let mut plans = Vec::new();
    let mut current_shape = TensorShape::from_chw(model.metadata.input_shape);

    for layer in &model.layers {
        match layer {
            Layer::Conv2d(conv) => {
                let in_shape = current_shape.clone();
                let out_shape = in_shape.after_conv2d(conv.out_channels, conv.kernel_size);

                let tiles_x = (conv.kernel_size + 3) / 4;
                let tiles_y = (conv.kernel_size + 1) / 2;
                let tiles_per_kernel = tiles_x * tiles_y;

                plans.push(LayerPlan::Conv2d {
                    name: conv.name.clone(),
                    in_shape: in_shape.clone(),
                    out_shape: out_shape.clone(),
                    num_convolve_ops: conv.out_channels * conv.in_channels * tiles_per_kernel,
                    num_accumulate_ops: conv.out_channels,
                });

                current_shape = out_shape;
            }

            Layer::Linear(linear) => {
                plans.push(LayerPlan::Linear {
                    name: linear.name.clone(),
                    in_size: linear.in_features,
                    out_size: linear.out_features,
                    num_mul_acc_ops: linear.out_features,
                });

                current_shape = TensorShape::Vector {
                    size: linear.out_features,
                };
            }

            Layer::MaxPool2d(pool) => {
                let in_shape = current_shape.clone();
                let out_shape = in_shape.after_pool(pool.kernel_size, pool.stride);

                plans.push(LayerPlan::MaxPool2d {
                    name: pool.name.clone(),
                    in_shape,
                    out_shape: out_shape.clone(),
                });

                current_shape = out_shape;
            }

            Layer::AvgPool2d(pool) => {
                let in_shape = current_shape.clone();
                let out_shape = in_shape.after_pool(pool.kernel_size, pool.stride);

                plans.push(LayerPlan::AvgPool2d {
                    name: pool.name.clone(),
                    in_shape,
                    out_shape: out_shape.clone(),
                });

                current_shape = out_shape;
            }

            Layer::Flatten(flat) => {
                let in_shape = current_shape.clone();
                let out_shape = in_shape.after_flatten();
                let out_size = out_shape.as_vector().unwrap_or(0);

                plans.push(LayerPlan::Flatten {
                    name: flat.name.clone(),
                    in_shape,
                    out_size,
                });

                current_shape = out_shape;
            }
        }
    }

    plans
}
