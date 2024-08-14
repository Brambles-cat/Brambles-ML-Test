package com.brambles;

import java.util.function.Function;

public class Stuff {
    public static class M_Double {
        public double value;
        public M_Double(double value) {
            this.value = value;
        }
    }

    public static Dataset circleDataSet(int exampleCount, double radius, double secondGroupDistance, double noise) {
        Dataset dataset = new Dataset(exampleCount, 2, 1);

        for (int i = 0; i < exampleCount; ++i) {
            if (Math.random() < 0.5) {
                // inner data points
                double r = Math.random() * radius + Math.random() * noise;
                double theta = Math.random() * 2 * Math.PI;
                dataset.examples[i][0] = r * Math.sin(theta);
                dataset.examples[i][1] = r * Math.cos(theta);
                dataset.labels[i][0] = 0;
                continue;
            }

            // outer data points
            double r = radius + secondGroupDistance + Math.random() * radius / 2f + Math.random() * noise;
            double theta = Math.random() * 2 * Math.PI;
            dataset.examples[i][0] = r * Math.sin(theta);
            dataset.examples[i][1] = r * Math.cos(theta);
            dataset.labels[i][0] = 1;
        }

        return dataset;
    }

    public static class Dataset {
        public double[][] examples, labels;

        public Dataset(int exampleCount, int featureCount, int labelCount) {
            examples = new double[exampleCount][featureCount];
            labels   = new double[exampleCount][labelCount];
        }

        public Dataset(double[][] examples, double[][] labels) {
            this.examples = examples;
            this.labels   = labels;
        }
    }

    public static Dataset lineDataSet(int exampleCount, double m, double b, double noise) {
        Dataset dataset = new Dataset(exampleCount, 1, 1);

        for (int i = 0; i < exampleCount; ++i) {
            dataset.examples[i][0] = i;
            dataset.labels[i][0] = m * i + b + noise * Math.random();
        }

        return dataset;
    }

    public static Function<Double, Double> derivative(Function<Double, Double> func) {
        double zero_limit = 1e-7;
        return (x) -> (func.apply(x + zero_limit) - func.apply(x)) / zero_limit;
    }
}
