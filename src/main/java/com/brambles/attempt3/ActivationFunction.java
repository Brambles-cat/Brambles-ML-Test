package com.brambles.attempt3;

import java.util.function.Function;

public enum ActivationFunction {
    LINEAR(sum -> sum, sum -> 1d),
    SIGMOID(
            sum -> 1d / (1d + Math.pow(Math.E, -sum)),
            sum -> {
                double activation = 1d / (1d + Math.pow(Math.E, -sum));
                return activation * (1 - activation);
            }
    ),
    RELU(
            sum -> Math.max(0, sum),
            sum -> sum == Math.max(0, sum) ? 1d : 0d
    );

    private final Function<Double, Double> activation, derivative;

    ActivationFunction(Function<Double, Double> activation, Function<Double, Double> derivative) {
        this.activation = activation;
        this.derivative = derivative;
    }

    public double apply(double weighted_sum) {
        return activation.apply(weighted_sum);
    }
    public double derivApply(double weighted_sum) {
        return derivative.apply(weighted_sum);
    }
}