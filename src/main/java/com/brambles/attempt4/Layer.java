package com.brambles.attempt4;

import java.util.Arrays;

public class Layer {
    Layer nextLayer;
    public ActivationFunction activation;
    public int inputCount, nodeCount;

    double[][] weightCostGradients;
    public double[][] weights;
    public double[]
            biases,
            biasCostGradients,
            weightedSums,
            inputs,
            activations;



    public Layer(int inputCount, int nodeCount, ActivationFunction activation) {
        weights             = new double[nodeCount][inputCount];
        weightCostGradients = new double[nodeCount][inputCount];
        biases              = new double[nodeCount];
        biasCostGradients   = new double[nodeCount];
        weightedSums        = new double[nodeCount];
        activations         = new double[nodeCount];
        inputs              = new double[inputCount];

        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; ++j)
                weights[i][j] = Math.random();

            biases[i] = Math.random();
        }

        this.nodeCount    = nodeCount;
        this.inputCount   = inputCount;
        this.activation   = activation;
    }

    public void applyGradients(double learnRate) {
        for (int i_node = 0; i_node < weights.length; ++i_node) {
            biases[i_node] -= biasCostGradients[i_node] * learnRate;

            for (int i_weight = 0; i_weight <  inputCount; ++i_weight)
                weights[i_node][i_weight] -= weightCostGradients[i_node][i_weight] * learnRate;
        }
    }

    double[] forward(double[] inputs) {
        this.inputs = inputs;

        for (int i_node = 0; i_node < nodeCount; ++i_node) {
            double weighted_sum = biases[i_node];

            for (int i_input = 0; i_input < inputs.length; i_input++)
                weighted_sum += weights[i_node][i_input] * inputs[i_input];

            weightedSums[i_node] = weighted_sum;
        }

        for (int i = 0; i < nodeCount; i++)
            activations[i] = activation.apply(weightedSums, i);

        return nextLayer != null ? nextLayer.forward(activations) : activations;
    }

    public double[] evalForward(double[] inputs) {
        double[] layerSums = new double[nodeCount];

        for (int i_node = 0; i_node < nodeCount; ++i_node) {
            double weighted_sum = biases[i_node];

            for (int i_input = 0; i_input < inputs.length; i_input++)
                weighted_sum += weights[i_node][i_input] * inputs[i_input];

            layerSums[i_node] = weighted_sum;
        }

        double[] layerActivations = new double[nodeCount];

        for (int i = 0; i < nodeCount; i++)
            layerActivations[i] = activation.apply(layerSums, i);

        return nextLayer != null ? nextLayer.evalForward(layerActivations) : layerActivations;
    }

    public void clearGradients() {
        for (int i_node = 0; i_node < nodeCount; i_node++) {
            Arrays.fill(weightCostGradients[i_node], 0);

            biasCostGradients[i_node] = 0;
        }
    }

    /**
     * @param nextLayersCostDerivs the cost derivatives with respect to the next layers' node's weighted sums
     * @return (dc/ds)[], the cost derivatives with respect to each node's weighted sums in this layer
     */
    public double[] costDerivStep(double[] nextLayersCostDerivs) {
        double[] costDerivStep = new double[nodeCount];

        for (int i_node = 0; i_node < nodeCount; ++i_node) {
            // dc/ds * ds/da = dc/da, ds/da = the weight that the input activation is multiplied by
            // Since each node can affect the cost through several paths to the next layer, make
            // them the sum of a costDerivStep * the respective weight from the following layer
            for (int i_nextLayerNode = 0; i_nextLayerNode < nextLayer.nodeCount; ++i_nextLayerNode)
                costDerivStep[i_node] += nextLayersCostDerivs[i_nextLayerNode] * nextLayer.weights[i_nextLayerNode][i_node];

            // dc/da * da/ds = dc/ds
            costDerivStep[i_node] *= activation.derivApply(weightedSums, i_node);
        }

        return costDerivStep;
    }

    /**
     * @param costDerivStep (dc/ds)[] the derivative of the output nodes cost with
     * respect to each node of the current layer's weighted sum
     */
    public void updateGradients(double[] costDerivStep) {
        for (int i_node = 0; i_node < nodeCount; ++i_node) {
            for (int i_inputNode = 0; i_inputNode < inputCount; ++i_inputNode) {
                // dc/ds * ds/dw = dc/dw, ds/dw = the activation from the previous layer multiplied by w
                weightCostGradients[i_node][i_inputNode] += costDerivStep[i_node] * inputs[i_inputNode];
            }

            // dc/ds * ds/db = dc/db, ds/db = 1
            biasCostGradients[i_node] += costDerivStep[i_node];
        }
    }

    /**
     * @param batchSize the amount of examples used in this training iteration to average the gradients with
     */
    public void averageGradients(int batchSize) {
        for (int i_node = 0; i_node < nodeCount; ++i_node) {
            for (int i_inputNode = 0; i_inputNode < inputCount; ++i_inputNode)
                weightCostGradients[i_node][i_inputNode] /= batchSize;

            biasCostGradients[i_node] /= batchSize;
        }
    }
}