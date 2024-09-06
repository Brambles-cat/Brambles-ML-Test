package com.brambles.attempt4;

import com.brambles.InferenceModel;
import com.brambles.Plotter;

import java.util.Arrays;

import static com.brambles.Stuff.Dataset;
import static com.brambles.Stuff.circleDataSet;

public class M4 {
    public static void main(String[] args) {
        // Create datasets with set seeds to compare with attempt3 while debugging
        Dataset trainingData = circleDataSet(12345, 150, 2.5, 0.5, 1);
        Dataset testingData  = circleDataSet(12345, 500, 2.5, 0.5, 1);

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
                0.03f,
                LossFunction.BCE,
                new Layer(2, 1, ActivationFunction.SIGMOID)
        )
                .addLayer(1, ActivationFunction.SIGMOID);

        // Split the datasets into batches
        for (int i = 0; i < batches.length; ++i) {
            Dataset batch = new Dataset(
                    Arrays.copyOfRange(trainingData.examples, i * batch_size, (i + 1) * batch_size),
                    Arrays.copyOfRange(trainingData.labels, i * batch_size, (i + 1) * batch_size)
            );

            batches[i] = batch;
        }

        System.out.println("Training in " + batches.length + " batches per epoch for " + epochs + " epochs");

        long timeStart = System.currentTimeMillis() / 1000;

        // Training loop
        for (int epoch = 1; epoch <= epochs; ++epoch) {
            for (Dataset batch : batches)
                nn.train(batch.examples, batch.labels);

            System.out.printf("\rProgress: %.2f", epoch * 100.0 / epochs);
        }

        // Results
        System.out.println("\n\nTraining finished in " + ((System.currentTimeMillis() / 1000) - timeStart) + " seconds");
        System.out.printf("\nTraining Cost: %.2f", nn.calculateCost(trainingData.examples, trainingData.labels));
        System.out.printf("\nTesting Cost: %.2f", nn.calculateCost(testingData.examples, testingData.labels));

        int correctly_classified = 0;
        for (int i = 0; i < testingData.examples.length; ++i) {

            double[] prediction = nn.execute(testingData.examples[i]);

            correctly_classified += Math.abs(prediction[0] - testingData.labels[i][0]) < 0.5 ? 1 : 0;
        }

        System.out.println("\n\nAccuracy: " + correctly_classified + " / " + testingData.examples.length + " (%" + correctly_classified * 100.0 / testingData.examples.length + ")");

        // Test loading and saving of the model
        nn.saveModel("outputs/circleClassifier.json");

        InferenceModel inferenceModel = InferenceModel.loadModel("outputs/circleClassifier.json");

        correctly_classified = 0;
        for (int i = 0; i < testingData.examples.length; ++i) {

            double[] prediction = inferenceModel.infer(testingData.examples[i]);

            correctly_classified += Math.abs(prediction[0] - testingData.labels[i][0]) < 0.5 ? 1 : 0;
        }

        System.out.println("\nLoaded model Accuracy: " + correctly_classified + " / " + testingData.examples.length + " (%" + correctly_classified * 100.0 / testingData.examples.length + ")");
    }
}
