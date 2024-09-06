package com.brambles.attempt2;

import com.brambles.Stuff;

import java.util.*;
import java.util.function.Function;

import static com.brambles.Stuff.*;

class NeuralNetwork {
    int inputCount;
    List<Layer> layers = new ArrayList<>();
    LossFunction lossFunction;
    double learningRate;

    public NeuralNetwork(double learning_rate, LossFunction loss_function, int inputCount) {
        learningRate = learning_rate;
        lossFunction = loss_function;
        this.inputCount = inputCount;
    }

    public Output execute(double[] inputs, Double[] labels, boolean train) {
        assert (!layers.isEmpty());
        assert (inputs.length == inputCount);

        Layer firstLayer = layers.getFirst();
        for (int i_input = 0; i_input < inputCount; ++i_input)
            for (Neuron neuron : firstLayer.nodes)
                neuron.inputs[i_input] = inputs[i_input];

        double[] outputs = firstLayer.forward();
        double[] losses = new double[outputs.length];

        for (int i = 0; i < outputs.length; ++i)
            losses[i] = lossFunction.apply(labels[i], outputs[i]);

        if (train) train(labels);

        return new Output(outputs, losses);
    }

    public void addLayer(int nodeCount, ActivationFunction activationFunction) {
        if (layers.isEmpty()) {
            Layer newLayer = new Layer(nodeCount, inputCount, activationFunction);
            layers.add(newLayer);
            return;
        }

        Layer newLayer = new Layer(nodeCount, layers.getLast().nodes.size(), activationFunction);
        layers.getLast().nextLayer = newLayer;
        layers.add(newLayer);
    }

    private void train(Double[] labels) {
        double[][] previousNudges = layers.getLast().calculateNudges(labels, lossFunction, learningRate);
        Double[] desiredOutputs;
        Layer currentLayer;
        
        for (int i = layers.size() - 2; i >= 0; --i) {
            currentLayer = layers.get(i);
            desiredOutputs = currentLayer.nodes.stream().map(Neuron::activate).toArray(Double[]::new);

            for (int i_neuron = 0; i_neuron < previousNudges.length; i_neuron++) {
                for (int i_nudge = 0; i_nudge < previousNudges[i_neuron].length ; i_nudge++) {
                    desiredOutputs[i_neuron] += previousNudges[i_neuron][i_nudge];
                }
            }
            previousNudges = currentLayer.calculateNudges(desiredOutputs, lossFunction, learningRate);
        }
    }

    static class Output {
        double[] outputs, losses;

        public Output(double[] outputs, double[] losses) {
            this.outputs = outputs;
            this.losses = losses;
        }
    }
}

class Neuron {
    Double[] weights;
    double[] inputs;
    final Function<Double, Double>[] weight_specific_activations;
    double bias = 0;
    final Function<Double, Double> bias_activation;
    final ActivationFunction activation;
    double output;

    public Neuron(int inputCount, ActivationFunction activation) {
        weights = new Double[inputCount];
        Arrays.fill(weights, 0d);

        inputs = new double[inputCount];
        this.activation = activation;
        weight_specific_activations = new Function[weights.length];

        for (int i = 0; i < weight_specific_activations.length; ++i) {
            int index = i;
            weight_specific_activations[i] = (w) -> activation.apply(w * inputs[index] + getWeightedSum() - weights[index] * inputs[index]);
        }

        bias_activation = (b) -> activation.apply(b + getWeightedSum() - bias);
    }

    private double getWeightedSum() {
        double weighted_sum = bias;

        for (int i = 0; i < inputs.length; i++)
            weighted_sum += weights[i] * inputs[i];

        return weighted_sum;
    }

    public double activate() {
        double weighted_sum = getWeightedSum();
        output = activation.apply(weighted_sum);
        return output;
    }
}


public class M2 {
    public static void main(String[] args) {
        Dataset trainingData = circleDataSet(0, 150, 2.5, 1, 0);
        Dataset testingData  = circleDataSet(0, 50, 2.5, 1, 0);

        NeuralNetwork nn = new NeuralNetwork(
                0.03f,
                LossFunction.BINARY_CROSS_ENTROPY,
                2
        );

        nn.addLayer(1, ActivationFunction.SIGMOID);

        for (int epoch = 0; epoch < 1000; ++epoch) {
            for (double[] example : trainingData.examples) {
                nn.execute(new double[]{Math.pow(example[0], 2), Math.pow(example[1], 2)}, new Double[]{example[2]}, true);
            }
        }

        System.out.print("Accuracy: ");

        int correctly_classified = 0;
        for (double[] example : testingData.labels) {
            double label = example[2];
            double prediction = nn.execute(new double[] {Math.pow(example[0], 2), Math.pow(example[1], 2)}, new Double[]{label}, false).outputs[0];

            correctly_classified += Math.abs(prediction - label) < 0.5 ? 1 : 0;
        }

        System.out.println(correctly_classified / (double) testingData.examples.length);
    }
}
