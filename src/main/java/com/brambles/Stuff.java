package com.brambles;

import java.util.function.Function;

public class Stuff {
    static class M_Double {
        double value;
        M_Double(double value) {
            this.value = value;
        }
    }

    static Function<Double, Double> derivative(Function<Double, Double> func) {
        double zero_limit = 1e-6;
        return (x) -> (func.apply(x + zero_limit) - func.apply(x)) / zero_limit;
    }
}