package com.brambles.attempt2;

import java.util.function.Function;

public enum ActivationFunction {
    LINEAR(sum -> sum),
    SIGMOID(sum -> (1.0 / (1.0 + Math.pow(Math.E, -sum)))),
    RELU(sum -> Math.max(0, sum));

    private final Function<Double, Double> activation;

    ActivationFunction(Function<Double, Double> activation) {
        this.activation = activation;
    }

    public double apply(double weighted_sum) {
        return activation.apply(weighted_sum);
    }
}