package com.brambles.attempt4;

import java.util.function.BiFunction;
import static java.lang.Math.*;

public enum LossFunction {
    BCE(
            (outputs, desiredOutputs) ->  -(desiredOutputs[0] * log(outputs[0]) + (1 - desiredOutputs[0]) * log(1 - outputs[0])),
            (output, desiredOutput) -> -(desiredOutput / output) + (1 - desiredOutput) / (1 - output)
    ),
    CE(
            // Precondition is that desiredOutputs is an array with one 1 and the rest 0s
            (outputs, desiredOutputs) -> {
                int i = 0;

                for (; i < desiredOutputs.length; ++i) {
                    if (desiredOutputs[i] == 1)
                        break;
                    assert  desiredOutputs[i] == 0;
                }

                assert desiredOutputs[i] == 1;

                return Double.isNaN(log(outputs[i])) ? 0 : -log(outputs[i]);
            },
            (output, desiredOutput) -> {
                if (desiredOutput == 0 || output == 0)
                    return 0d;

                return -1 / output;
            }
    );

    private final BiFunction<double[], double[], Double> loss_fn;
    private final BiFunction<Double, Double, Double> loss_deriv;

    LossFunction(BiFunction<double[], double[], Double> loss_fn, BiFunction<Double, Double, Double> loss_deriv) {
        this.loss_fn = loss_fn;
        this.loss_deriv = loss_deriv;
    }

    double apply(double[] outputs, double[] desiredOutputs) {
        return loss_fn.apply(outputs, desiredOutputs);
    }

    double derivApply(double output, double desiredOutput) {
        return loss_deriv.apply(output, desiredOutput);
    }
}
