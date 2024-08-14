package com.brambles.attempt3;

import java.util.function.BiFunction;

public enum LossFunction {
    BCE(
            (prediction, expectedOutput) ->  -(expectedOutput * Math.log(prediction) + (1 - expectedOutput) * Math.log(1 - prediction)),
            (prediction, expectedOutput) -> -(expectedOutput / prediction) + (1 - expectedOutput) / (1 - prediction)
    ),
    SE(
            (prediction, expectedOutput) -> Math.pow(prediction - expectedOutput, 2),
            (prediction, expectedOutput) -> 2 * (prediction - expectedOutput)
    );

    private final BiFunction<Double, Double, Double> loss_fn, loss_deriv;

    LossFunction(BiFunction<Double, Double, Double> loss_fn, BiFunction<Double, Double, Double> loss_deriv) {
        this.loss_fn = loss_fn;
        this.loss_deriv = loss_deriv;
    }

    double apply(double prediction, double expectedOutput) {
        return loss_fn.apply(prediction, expectedOutput);
    }

    double derivApply(double predictions, double expectedOutput) {
        return loss_deriv.apply(predictions, expectedOutput);
    }
}
