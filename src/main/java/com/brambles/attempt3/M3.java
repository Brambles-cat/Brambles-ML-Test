package com.brambles.attempt3;

import com.brambles.Plotter;

import java.util.Arrays;

import static com.brambles.Stuff.*;

public class M3 {
    public static void main(String[] args) {
        Dataset trainingData = circleDataSet(150, 2.5, 0.5, 1);
        Dataset testingData  = circleDataSet(500, 2.5, 0.5, 1);

        Plotter.plot(trainingData, 450);

        for (double[] example : trainingData.examples) {
            example[0] *= example[0];
            example[1] *= example[1];
        }
        for (double[] example : testingData.examples) {
            example[0] *= example[0];
            example[1] *= example[1];
        }

        int epochs = 10000, batch_size = 25;
        assert trainingData.examples.length % batch_size == 0;
        Dataset[] batches = new Dataset[trainingData.examples.length / batch_size];

        NeuralNetwork nn = new NeuralNetwork(
                0.03f, // / batches.length,
                LossFunction.BCE,
                new Layer(2, 1, ActivationFunction.SIGMOID)
        )
                .addLayer(1, ActivationFunction.SIGMOID);

        for (int i = 0; i < batches.length; ++i) {
            Dataset batch = new Dataset(
                    Arrays.copyOfRange(trainingData.examples, i * batch_size, (i + 1) * batch_size),
                    Arrays.copyOfRange(trainingData.labels, i * batch_size, (i + 1) * batch_size)
            );

            batches[i] = batch;
        }

        System.out.println("Training in " + batches.length + " batches per epoch");

        long timeStart = System.currentTimeMillis() / 1000;

        for (int epoch = 0; epoch < epochs; ++epoch) {
            for (Dataset batch : batches)
                nn.train(batch.examples, batch.labels);

            System.out.printf("\rProgress: %.2f", epoch * 100.0 / epochs);
        }

        System.out.println("\n\nTraining finished in " + ((System.currentTimeMillis() / 1000) - timeStart) + " seconds");
        System.out.printf("\nTraining Cost: %.2f", nn.calculateCost(trainingData.examples, trainingData.labels));
        System.out.printf("\nTesting Cost: %.2f", nn.calculateCost(testingData.examples, testingData.labels));
        System.out.print("\n\nAccuracy: ");

        int correctly_classified = 0;
        for (int i = 0; i < testingData.examples.length; ++i) {

            double[] prediction = nn.execute(testingData.examples[i]);

            correctly_classified += Math.abs(prediction[0] - testingData.labels[i][0]) < 0.5 ? 1 : 0;
        }

        System.out.println(correctly_classified + " / " + testingData.examples.length + " (%" + correctly_classified * 100.0 / testingData.examples.length + ")");
    }
}
