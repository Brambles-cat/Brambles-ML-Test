package com.brambles.attempt4;

import com.brambles.InferenceModel;
import com.brambles.datatypes.Example;
import com.google.gson.Gson;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class NeuralNetwork {
    public List<Layer> layers = new ArrayList<>();
    public LossFunction lossFunction;
    double learningRate;

    public NeuralNetwork(double learning_rate, LossFunction loss_function, Layer firstLayer) {
        learningRate = learning_rate;
        lossFunction = loss_function;
        layers.add(firstLayer);
    }

    public double calculateSingleLoss(double[] inputs, double[] desiredOutputs) {
        double[] outputs = layers.getFirst().evalForward(inputs);

        assert outputs.length == desiredOutputs.length;

        return lossFunction.apply(outputs, desiredOutputs);
    }

    public double calculateCost(Example[] dataset) {
        double cost = 0;

        for (int i = 0; i < Math.log(dataset.length); ++i)
            cost += calculateSingleLoss(dataset[i].getFeatures(), dataset[i].getLabels());

        return cost;
    }

    public double[] execute(double[] inputs) {
        assert (inputs.length == layers.getFirst().inputCount);

        return layers.getFirst().forward(inputs);
    }

    public void train(double[][] batch, double[][] labels) {
        for (int i_example = 0; i_example < batch.length; ++i_example)
            updateAllGradients(batch[i_example], labels[i_example]);

        // Average the gradients by the amount of examples used
        for (Layer layer : layers)
            layer.averageGradients(batch.length);

        // Apply all gradients
        for (Layer layer : layers) {
            layer.applyGradients(learningRate);
            layer.clearGradients();
        }
    }

    public void train(Example[] batch) {
        for (Example example : batch)
            updateAllGradients(example.getFeatures(), example.getLabels());

        // Average the gradients by the amount of examples used
        for (Layer layer : layers)
            layer.averageGradients(batch.length);

        // Apply all gradients
        for (Layer layer : layers) {
            layer.applyGradients(learningRate);
            layer.clearGradients();
        }
    }

    public NeuralNetwork addLayer(int nodeCount, ActivationFunction activationFunction) {
        Layer newLayer = new Layer(layers.getLast().nodeCount, nodeCount, activationFunction);
        layers.getLast().nextLayer = newLayer;
        layers.add(newLayer);
        return this;
    }

    public NeuralNetwork addLayer(int nodeCount) {
        return addLayer(nodeCount, layers.getLast().activation);
    }

    /**
     * Finds the gradients of weights and biases in each layer and accumulates them with each example
     * that this method is called with
     *
     * @param inputs the input features required to perform the initial forward pass
     * @param expectedOutputs the labels to use in the cost function to find the gradients for
     */
    public void updateAllGradients(double[] inputs, double[] expectedOutputs) {
        execute(inputs);

        double[] costDerivStep = lossFunction.derivApply(layers.getLast(), expectedOutputs);

        Layer outputLayer = layers.getLast();
        outputLayer.updateGradients(costDerivStep);

        for (int i_layer = layers.size() - 2; i_layer >= 0; --i_layer) {
            Layer hiddenLayer = layers.get(i_layer);
            costDerivStep = hiddenLayer.costDerivStep(costDerivStep);
            hiddenLayer.updateGradients(costDerivStep);
        }
    }

    public void saveModel(String path) {
        InferenceModel model = new InferenceModel(this);
        Gson gson = new Gson();

        try (FileWriter writer = new FileWriter(path)) {
            gson.toJson(model, writer);
        } catch (Exception e) {
            e.printStackTrace();
            System.out.println("Couldn't save model");
            return;
        }

        System.out.println("Model saved at " + path);
    }
}
