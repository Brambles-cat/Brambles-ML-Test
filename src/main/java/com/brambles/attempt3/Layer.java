package com.brambles.attempt3;

public class Layer {
    Layer nextLayer;
    ActivationFunction activation;
    int inputCount, nodeCount;

    double[][] weightCostGradients;
    double[][] weights;
    double[]
            biasCostGradients,
            biases,
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

        this.nodeCount    = nodeCount;
        this.inputCount   = inputCount;
        this.activation   = activation;
    }

    public void applyGradients(double learnRate) {
        for (int i_node = 0; i_node < weights.length; ++i_node) {
            biases[i_node] -= biasCostGradients[i_node] * learnRate;

            for (int i_weight = 0; i_weight < inputCount; ++i_weight)
                weights[i_node][i_weight] -= weightCostGradients[i_node][i_weight] * learnRate;
        }
    }

    double[] forward(double[] inputs) {
        this.inputs = inputs;

        for (int i_node = 0; i_node < biases.length; ++i_node) {
            double weighted_sum = biases[i_node];

            for (int i_input = 0; i_input < inputs.length; i_input++)
                weighted_sum += weights[i_node][i_input] * inputs[i_input];

            weightedSums[i_node] = weighted_sum;
            activations[i_node] = activation.apply(weighted_sum);
        }

        return nextLayer != null ? nextLayer.forward(activations) : activations;
    }
    
    public void updateGradients(double[] partialCostDerivs) {
        for (int i_nodeOut = 0; i_nodeOut < partialCostDerivs.length; ++i_nodeOut) {
            for (int i_input = 0; i_input < inputCount; ++i_input) {
                // dc/dz * dz/dw (the activation being used as an input here) = dc/dw
                double costDerivWrtWeight = inputs[i_input] * partialCostDerivs[i_nodeOut];
                weightCostGradients[i_nodeOut][i_input] += costDerivWrtWeight;
            }
            // dz/db = 1
            biasCostGradients[i_nodeOut] += partialCostDerivs[i_nodeOut];
        }
    }

    public double[] calculateHiddenLayerPartialCostDerivs(Layer followingLayer, double[] followingLayerPartialCostDeriv) {
        double[] newPartialCostDerivs = new double[nodeCount];

        for (int i_node = 0; i_node < followingLayer.nodeCount; ++i_node) {
            double newPartialCostDeriv = 0;

            for (int i_followingDeriv = 0; i_followingDeriv < followingLayerPartialCostDeriv.length; ++i_followingDeriv) {
                double weightedInputDeriv = followingLayer.weights[i_node][i_followingDeriv];
                newPartialCostDeriv += weightedInputDeriv * followingLayerPartialCostDeriv[i_followingDeriv];
            }

            newPartialCostDeriv *= activation.derivApply(weightedSums[i_node]);
            newPartialCostDerivs[i_node] = newPartialCostDeriv;
        }

        return newPartialCostDerivs;
    }

    public void clearGradients() {
        for (int i_node = 0; i_node < nodeCount; i_node++) {
            for (int i_weight = 0; i_weight < weightCostGradients[i_node].length; ++i_weight)
                weightCostGradients[i_node][i_weight] = 0;

            biasCostGradients[i_node] = 0;
        }
    }
}