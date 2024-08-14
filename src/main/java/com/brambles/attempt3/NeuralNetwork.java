package com.brambles.attempt3;

import java.util.ArrayList;
import java.util.List;

public class NeuralNetwork {
    List<Layer> layers = new ArrayList<>();
    LossFunction lossFunction;
    double learningRate;

    public NeuralNetwork(double learning_rate, LossFunction loss_function, Layer firstLayer) {
        learningRate = learning_rate;
        lossFunction = loss_function;
        layers.add(firstLayer);
    }

    public double calculateSingleLoss(double[] inputs, double[] desiredOutputs) {
        double[] outputs = layers.getFirst().forward(inputs);

        assert outputs.length == desiredOutputs.length;

        double cost = 0;

        for (int i = 0; i < desiredOutputs.length; ++i)
            cost += lossFunction.apply(outputs[i], desiredOutputs[i]);

        return cost;
    }

    public double calculateCost(double[][] dataset, double[][] labels) {
        double cost = 0;

        for (int i = 0; i < dataset.length; ++i)
            cost += calculateSingleLoss(dataset[i], labels[i]);

        return cost / dataset.length;
    }

    public double[] execute(double[] inputs) {
        assert (inputs.length == layers.getFirst().inputCount);

        return layers.getFirst().forward(inputs);
    }

    public void train(double[][] batch, double[][] labels) {
        for (int i_example = 0; i_example < batch.length; ++i_example)
            updateAllGradients(batch[i_example], labels[i_example]);

        // Apply all gradients
        for (Layer layer : layers) {
            layer.applyGradients(learningRate);
            layer.clearGradients();
        }
    }

    public double[] partialCostDeriv(double[] expectedOutputs) {
        double[] partialDerivs = new double[expectedOutputs.length];
        Layer outputLayer = layers.getLast();

        for (int i = 0; i < partialDerivs.length; ++i) {
            // Find how much the cost changes by the change in the activation and how much the activation changes by a change in the weightedInputs
            double costDeriv = lossFunction.derivApply(outputLayer.activations[i], expectedOutputs[i]);
            double activationDeriv = outputLayer.activation.derivApply(outputLayer.weightedSums[i]);
            partialDerivs[i] = activationDeriv * costDeriv;
        }

        // Returning (dc/dz)[], where z is a weighted sum being fed into an output layer node
        return partialDerivs;
    }

    public NeuralNetwork addLayer(int nodeCount, ActivationFunction activationFunction) {
        Layer newLayer = new Layer(layers.getLast().biases.length, nodeCount, activationFunction);
        layers.getLast().nextLayer = newLayer;
        layers.add(newLayer);
        return this;
    }

    public void updateAllGradients(double[] inputs, double[] expectedOutputs) {
        execute(inputs);

        double[] partialCostDerivs = partialCostDeriv(expectedOutputs);
        Layer outputLayer = layers.getLast();
        outputLayer.updateGradients(partialCostDerivs);

        for (int i_layer = layers.size() - 2; i_layer >= 0; --i_layer) {
            Layer hiddenLayer = layers.get(i_layer);
            partialCostDerivs = hiddenLayer.calculateHiddenLayerPartialCostDerivs(hiddenLayer.nextLayer, partialCostDerivs);
            hiddenLayer.updateGradients(partialCostDerivs);
        }

    }
}
