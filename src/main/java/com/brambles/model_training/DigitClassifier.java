package com.brambles.model_training;

import com.brambles.attempt4.ActivationFunction;
import com.brambles.attempt4.Layer;
import com.brambles.attempt4.LossFunction;
import com.brambles.attempt4.NeuralNetwork;
import com.brambles.datatypes.GrayScaleImage;

import java.util.Arrays;

import static com.brambles.Stuff.batch;
import static com.brambles.Stuff.loadDataset;
import static com.brambles.InferenceModel.loadModel;

public class DigitClassifier {
    static final int
        batch_size = 1,
        epochs = 100;

    static final double learningRate = 0.03;

    static NeuralNetwork nn = null; // loadModel("outputs/digitClassifier.json").toTrainable(learningRate, LossFunction.BCE);

    public static void main(String[] args) {
        GrayScaleImage[]
                trainingData = (GrayScaleImage[]) loadDataset("outputs/train.json"),
                testingData = (GrayScaleImage[]) loadDataset("outputs/test.json");

        System.out.println("Datasets loaded");

        GrayScaleImage[][] trainingBatches = batch(trainingData, batch_size);

        if (nn == null)
            nn = new NeuralNetwork(learningRate, LossFunction.CE, new Layer(
                    28 * 28, 50, ActivationFunction.SIGMOID
            ))
                    .addLayer(20).addLayer(10, ActivationFunction.SOFT_MAX);

        double prevCost = nn.calculateCost(trainingData);

        for (int epoch = 1; epoch <= epochs; ++epoch) {
            if (epoch % 1 == 0) {
                double cost = nn.calculateCost(trainingData);
                System.out.printf("\rEpoch " + epoch + "/" + epochs + "\nTraining Cost: %.2f\t\t(" + (cost - prevCost) + ")\nTest Cost: %.2f\n\n", cost, nn.calculateCost(testingData));
                prevCost = cost;
            }

            for (GrayScaleImage[] batch : trainingBatches)
                nn.train(batch);

            System.out.printf("\rProgress: %.2f%%", epoch * 100.0 / epochs);
        }

        System.out.println();

        nn.saveModel("outputs/digitClassifier3.json");

        for (GrayScaleImage example : trainingBatches[0])
            System.out.println(Arrays.stream(nn.execute(example.getFeatures())).boxed().toList() + " - " + example.label);
    }
}
