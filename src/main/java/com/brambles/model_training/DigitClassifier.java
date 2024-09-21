package com.brambles.model_training;

import com.brambles.attempt4.ActivationFunction;
import com.brambles.attempt4.Layer;
import com.brambles.attempt4.LossFunction;
import com.brambles.attempt4.NeuralNetwork;
import com.brambles.datatypes.Example;
import com.brambles.datatypes.GrayScaleImage;

import static com.brambles.Stuff.batch;
import static com.brambles.Stuff.loadDataset;

public class DigitClassifier {
    static final int
        batch_size = 100,
        epochs = 25;

    static final double learningRate = 0.005;

    static NeuralNetwork nn = null; // loadModel("outputs/digitClassifier.json").toTrainable(learningRate, LossFunction.BCE);

    public static void main(String[] args) {
        GrayScaleImage[]
                trainingData = (GrayScaleImage[]) loadDataset("outputs/train.json"),
                testingData = (GrayScaleImage[]) loadDataset("outputs/test.json");

        System.out.println("Datasets loaded");

        GrayScaleImage[][] trainingBatches = batch(trainingData, batch_size);

        System.out.println("Training in " + trainingBatches.length + " batches of " + batch_size);

        if (nn == null)
            nn = new NeuralNetwork(learningRate, LossFunction.CROSS_ENTROPY, new Layer(
                    28 * 28, 50, ActivationFunction.SIGMOID
            ))
                    .addLayer(20, ActivationFunction.SIGMOID)
                    .addLayer(10, ActivationFunction.SOFT_MAX);

        double prevCost = nn.calculateCost(trainingData);

        for (int epoch = 0; epoch < epochs; ++epoch) {
            if (epoch % 5 == 0) {
                double cost = nn.calculateCost(trainingData);
                printAccuracy(nn, trainingData);
                System.out.printf("\rEpoch " + epoch + "/" + epochs + "\nTraining Cost: %.2f\t\t(" + (cost - prevCost) + ")\nTest Cost: %.2f\n\n", cost, nn.calculateCost(testingData));
                prevCost = cost;
            }

            for (int i_batch = 0; i_batch < trainingBatches.length; i_batch++) {
                nn.train(trainingBatches[i_batch]);

                System.out.printf("\rProgress: %.2f%%", ((epoch + ((double) i_batch + 1) / trainingBatches.length)) * 100.0 / epochs);
            }
        }

        System.out.println();

        nn.saveModel("outputs/digitClassifier3.json");
    }


    static void printAccuracy(NeuralNetwork n, GrayScaleImage[] examples) {
        int correct = 0;

        for (Example example : examples) {
            double[] outputs = n.layers.getFirst().evalForward(example.getFeatures());

            int prediction = 0;
            for (int i = 1; i < outputs.length; i++) {
                if (outputs[i] > outputs[prediction])
                    prediction = i;
            }

            if (example.getLabels()[prediction] == 1)
                ++correct;
        }

        System.out.println("\n\nAccuracy: " + (double) correct / examples.length * 100);
    }


    // double[][] raw_image = new double[28][28];


        /*
        for (GrayScaleImage example : trainingData) {
            System.out.println(example.label);

            BufferedImage image = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);

            for (int i = 0; i < 28; i++)
                System.arraycopy(example.image, i * 28, raw_image[i], 0, 28);

            for (int x = 0; x < 28; x++) {
                for (int y = 0; y < 28; y++) {
                    // Ensure the value is between 0 and 1
                    float value = (float) Math.max(0, Math.min(1, raw_image[y][x]));
                    // Convert the float value to a grayscale value (0-255)
                    int gray = (int) (value * 255);
                    int rgb = (gray << 16) | (gray << 8) | gray;
                    image.setRGB(x, y, rgb);
                }
            }

            // Save the image to a file
            try {
                ImageIO.write(image, "png", new File("grayscale_image.png"));
                image.flush();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        */
}
