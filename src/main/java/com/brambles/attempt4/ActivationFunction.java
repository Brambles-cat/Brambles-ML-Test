package com.brambles.attempt4;

import java.util.Arrays;
import java.util.function.BiFunction;

import static java.lang.Math.*;

public enum ActivationFunction {
    LINEAR((sums, index) -> sums[index], (sums, index) -> 1d),
    SIGMOID(
            (sums, index) -> 1d / (1d + pow(E, -sums[index])),
            (sums, index) -> {
                double activation = 1d / (1d + pow(E, -sums[index]));
                return activation * (1 - activation);
            }
    ),
    RELU(
            (sums, index) -> max(0, sums[index]),
            (sums, index) -> sums[index] == max(0, sums[index]) ? 1d : 0
    ),
    SOFT_MAX(
            (sums, index) -> {
                double max_logit_val = sums[0];

                for (int i = 1; i < sums.length; ++i) {
                    if (sums[i] > max_logit_val)
                        max_logit_val = sums[i];
                }

                for (int i = 0; i < sums.length; ++i)
                    sums[i] -= max_logit_val;

                double numerator = exp(sums[index]);
                double denominator = Arrays.stream(sums).map(Math::exp).sum();
                return numerator / denominator;
            },
            (sums, index) -> {
                double expSumsSum = Arrays.stream(sums).map(Math::exp).sum();
                return (exp(sums[index]) / expSumsSum) * (1 - exp(sums[index]) / expSumsSum);
            }
    );

    public final BiFunction<double[], Integer, Double> function, derivative;

    ActivationFunction(BiFunction<double[], Integer, Double> activation, BiFunction<double[], Integer, Double> derivative) {
        this.function = activation;
        this.derivative = derivative;
    }

    /**
     * @param layer_sums The weighted sums from all nodes in the current layer
     * @param node_index The index of the node to be activated
     */
    public double apply(double[] layer_sums, int node_index) {
        return function.apply(layer_sums, node_index);
    }

    /**
     * @param layer_sums The weighted sums from all nodes in the current layer
     * @param node_index The index of the node to be activated
     */
    public double derivApply(double[] layer_sums, int node_index) {
        return derivative.apply(layer_sums, node_index);
    }
}