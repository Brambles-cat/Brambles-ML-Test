package com.brambles.attempt4;

import java.util.function.BiFunction;
import static java.lang.Math.*;

public enum LossFunction {
    CROSS_ENTROPY(
        // Precondition is that desiredOutputs is an array with one 1 and the rest 0s
        (outputs, desiredOutputs) -> {
            int i = 0;

            for (; i < desiredOutputs.length; ++i) {
                if (desiredOutputs[i] == 1)
                    break;

                assert desiredOutputs[i] == 0;
            }

            assert desiredOutputs[i] == 1;

            return Double.isNaN(log(outputs[i])) ? 0 : -log(outputs[i]);
        },
        (outputLayer, desiredOutputs) -> {
            double[] ret = new double[outputLayer.weightedSums.length];

            for (int i = 0; i < ret.length; ++i) {
                ret[i] = outputLayer.activations[i];

                if (desiredOutputs[i] == 1)
                    ret[i] -= 1;
            }

            return ret;
        }
    );

    private final BiFunction<double[], double[], Double> loss_fn;
    private final BiFunction<Layer, double[], double[]> loss_deriv;

    LossFunction(BiFunction<double[], double[], Double> loss_fn, BiFunction<Layer, double[], double[]> loss_deriv) {
        this.loss_fn = loss_fn;
        this.loss_deriv = loss_deriv;
    }

    double apply(double[] outputs, double[] desiredOutputs) {
        return loss_fn.apply(outputs, desiredOutputs);
    }

    double[] derivApply(Layer outputLayer, double[] desiredOutputs) {
        return loss_deriv.apply(outputLayer, desiredOutputs);
    }
}
