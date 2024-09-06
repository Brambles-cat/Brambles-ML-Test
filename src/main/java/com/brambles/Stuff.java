package com.brambles;

import com.brambles.datatypes.Example;
import com.brambles.datatypes.GrayScaleImage;
import com.google.gson.Gson;

import java.io.FileReader;
import java.io.IOException;
import java.lang.reflect.Array;
import java.util.Random;
import java.util.function.Function;

public class Stuff {
    public static class M_Double {
        public double value;
        public M_Double(double value) {
            this.value = value;
        }
    }

    public static Dataset circleDataSet(long seed, int exampleCount, double radius, double secondGroupDistance, double noise) {
        Dataset dataset = new Dataset(exampleCount, 2, 1);
        Random rand = new Random(seed);

        for (int i = 0; i < exampleCount; ++i) {
            if (rand.nextBoolean()) {
                // inner data points
                double r = rand.nextDouble() * radius + rand.nextDouble() * noise;
                double theta = rand.nextDouble() * 2 * Math.PI;
                dataset.examples[i][0] = r * Math.sin(theta);
                dataset.examples[i][1] = r * Math.cos(theta);
                dataset.labels[i][0] = 0;
                continue;
            }

            // outer data points
            double r = radius + secondGroupDistance + rand.nextDouble() * radius / 2f + rand.nextDouble() * noise;
            double theta = rand.nextDouble() * 2 * Math.PI;
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

    public static <T> T[][] batch(T[] dataset, int batch_size) {
        assert (dataset.length > 0);
        assert (dataset.length % batch_size == 0);

        int batches = dataset.length / batch_size;

        @SuppressWarnings("unchecked")
        T[][] dataset_batches = (T[][]) Array.newInstance(dataset[0].getClass(), batches, batch_size);

        for (int i_batch = 0; i_batch < batches; ++i_batch)
            System.arraycopy(dataset, i_batch * batch_size, dataset_batches[i_batch], 0, batch_size);

        return dataset_batches;
    }

    public static Example[] loadDataset(String path) {
        Gson g = new Gson();
        Example[] dataset;

        try (FileReader r = new FileReader(path)) {
            dataset = g.fromJson(r, GrayScaleImage[].class);
        }
        catch (IOException e) { throw new RuntimeException(e); }

        for (Example example : dataset)
            example.postDeserialization();

        return dataset;
    }
}
