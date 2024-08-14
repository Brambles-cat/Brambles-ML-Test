package com.brambles.attempt2;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

import static com.brambles.Stuff.derivative;

public class Layer {
    Layer nextLayer;

    List<Neuron> nodes = new ArrayList<>();
    public Layer(int nodeCount, int inputNodes, ActivationFunction activation) {
        for (int i = 0; i < nodeCount; ++i)
            nodes.add(new Neuron(inputNodes, activation));
    }

    double[] forward() {
        if (nextLayer == null) {
            double[] outputs = new double[nodes.size()];

            for (int i = 0; i < nodes.size(); ++i)
                outputs[i] = nodes.get(i).activate();

            return outputs;
        }

        for (int i = 0; i < nodes.size(); i++) {
            double output = nodes.get(i).activate();

            for (Neuron node : nextLayer.nodes)
                node.inputs[i] = output;
        }

        return nextLayer.forward();
    }

    public double[][] calculateNudges(Double[] desiredOutputs, LossFunction loss_fn, double learning_rate) {
        int weight_count = nodes.getFirst().inputs.length;
        double[][] nudges = new double[nodes.size()][weight_count + 1];

        Function<Double, Double> gradient;

        for (int i_neuron = 0; i_neuron < nudges.length; ++i_neuron) {
            Neuron n = nodes.get(i_neuron);
            double desiredOutput = desiredOutputs[i_neuron], nudge;

            for (int i_weight = 0; i_weight < weight_count; ++i_weight) {
                int ef_i_weight = i_weight;

                gradient = derivative((weight) -> loss_fn.apply(desiredOutput, n.weight_specific_activations[ef_i_weight].apply(weight)));

                nudge = -gradient.apply(n.weights[i_weight]);


                nudges[i_neuron][i_weight] = nudge;
                n.weights[i_weight] = n.weights[i_weight] + learning_rate * nudge;

            }

            gradient = derivative((bias) -> loss_fn.apply(desiredOutput, n.bias_activation.apply(bias)));
            nudge = -gradient.apply(n.bias);
            nudges[i_neuron][weight_count] = nudge;
            n.bias = n.bias + learning_rate * nudge;
        }

        return nudges;
    }
}