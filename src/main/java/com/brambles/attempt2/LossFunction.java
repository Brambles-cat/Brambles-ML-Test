package com.brambles.attempt2;

import java.util.function.BiFunction;

public enum LossFunction {
    BINARY_CROSS_ENTROPY((label, prediction) -> -(label * Math.log(prediction) + (1 - label) * Math.log(1 - prediction))),
    SQUARED_ERROR((label, prediction) -> Math.pow(label - prediction, 2));

    private final BiFunction<Double, Double, Double> loss_fn;

    LossFunction(BiFunction<Double, Double, Double> loss_fn) {
        this.loss_fn = loss_fn;
    }

    double apply(double label, double prediction) {
        return loss_fn.apply(label, prediction);
    }
}
