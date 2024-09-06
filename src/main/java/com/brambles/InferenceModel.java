package com.brambles;

import com.brambles.attempt4.ActivationFunction;
import com.brambles.attempt4.Layer;
import com.brambles.attempt4.LossFunction;
import com.brambles.attempt4.NeuralNetwork;
import com.google.gson.Gson;

import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.function.BiFunction;

public class InferenceModel {
    private transient BiFunction<double[], Integer, Double>[] activationFuncs;
    private final String[] activationNames;
    /**
     * Weights stored by [layer][node][weight] with biases being placed at the ends of the weight[] arrays
     */
    private final double[][][] weights_bias;

    /**
     * Convert a neural network into an optimized inference only model that
     * discludes the extra data caching and training steps
     *
     * @param neuralNetwork
     */
    @SuppressWarnings("unchecked")
    public InferenceModel(NeuralNetwork neuralNetwork) {
        activationNames = neuralNetwork.layers.stream().map(layer -> layer.activation.name()).toArray(String[]::new);
        weights_bias = new double[neuralNetwork.layers.size()][][];
        activationFuncs = new BiFunction[neuralNetwork.layers.size()];
        Layer layer;

        // Initialize the model by extracting the relevant nn values from its objects and into a basic 3d array
        for (int i_layer = 0; i_layer < neuralNetwork.layers.size(); ++i_layer) {
            layer = neuralNetwork.layers.get(i_layer);
            weights_bias[i_layer] = new double[layer.nodeCount][];
            activationFuncs[i_layer] = layer.activation.function;

            for (int i_node = 0; i_node < layer.nodeCount; ++i_node) {
                weights_bias[i_layer][i_node] = new double[layer.inputCount + 1];

                System.arraycopy(layer.weights[i_node], 0, weights_bias[i_layer][i_node], 0, layer.inputCount);

                weights_bias[i_layer][i_node][layer.inputCount] = layer.biases[i_node];
            }
        }
    }

    /**
     * @param inputs
     * @return the label inferences that the model is made to make accurate predictions for
     */
    public double[] infer(double[] inputs) {
        double[] outputs = null, sums;

        for (int i_layer = 0; i_layer < activationFuncs.length; ++i_layer) {
            sums = new double[weights_bias[i_layer].length];
            outputs = new double[weights_bias[i_layer].length];

            for (int i_node = 0; i_node < outputs.length; ++i_node) {
                for (int i_input = 0; i_input < inputs.length; ++i_input)
                    sums[i_node] += weights_bias[i_layer][i_node][i_input] * inputs[i_input];

                sums[i_node] += weights_bias[i_layer][i_node][inputs.length];
            }

            for (int i_node = 0; i_node < outputs.length; ++i_node)
                outputs[i_node] = activationFuncs[i_layer].apply(sums, i_node);

            inputs = outputs;
        }

        return outputs;
    }

    /**
     * Convert the inference model into neural network with training implementations
     *
     * @param learningRate
     * @param lossFunction
     * @return A trainable neural network
     */
    public NeuralNetwork toTrainable(double learningRate, LossFunction lossFunction) {
        Layer layer = new Layer(
                weights_bias[0][0].length - 1,
                weights_bias[0].length,
                ActivationFunction.valueOf(activationNames[0])
        );

        for (int i_node = 0; i_node < layer.nodeCount; ++i_node) {
            System.arraycopy(weights_bias[0][i_node], 0, layer.weights[i_node], 0, layer.inputCount);

            layer.biases[i_node] = weights_bias[0][i_node][layer.inputCount];
        }
        
        NeuralNetwork nn = new NeuralNetwork(
                learningRate,
                lossFunction,
                layer
        );

        for (int i_layer = 1; i_layer < weights_bias.length; i_layer++) {
            nn.addLayer(
                    weights_bias[i_layer].length,
                    com.brambles.attempt4.ActivationFunction.valueOf(activationNames[i_layer])
            );

            layer = nn.layers.getLast();

            for (int i_node = 0; i_node < layer.nodeCount; ++i_node) {
                System.arraycopy(weights_bias[i_layer][i_node], 0, layer.weights[i_node], 0, layer.inputCount);

                layer.biases[i_node] = weights_bias[i_layer][i_node][layer.inputCount];
            }
        }
        
        return nn;
    }

    /**
     * Load an inference model from a json file
     *
     * @param path path to the json file
     * @return an Inference only model
     */
    @SuppressWarnings("unchecked")
    public static InferenceModel loadModel(String path) {
        Gson gson = new Gson();

        try (FileReader reader = new FileReader(path)) {
            InferenceModel model = gson.fromJson(reader, InferenceModel.class);
            model.activationFuncs = Arrays.stream(model.activationNames).map(name -> ActivationFunction.valueOf(name).function).toArray(BiFunction[]::new);
            System.out.println("Model loaded from " + path);
            return model;
        }

        catch (IOException e) { throw new RuntimeException(e); }
    }
}
