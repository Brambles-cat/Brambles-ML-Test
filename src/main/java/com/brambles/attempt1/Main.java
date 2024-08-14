package com.brambles.attempt1;

import com.brambles.Stuff;

import java.util.function.Function;

import static com.brambles.Stuff.*;

public class Main {

    public static void main(String[] args) {
        M_Double w = new M_Double(1.0), b = new M_Double(1.0), leaning_rate = new M_Double(0.00005);
        Function<Double, Double> clouds_to_rain = (x) -> w.value * x + b.value;

        double[][] rClouds_cRain = new double[20][2];

        for (int i = 0; i < rClouds_cRain.length; ++i) {
            rClouds_cRain[i][0] = Math.random() * 100;
            // Ignore this. Just assume it constantly rains at least 20 units
            rClouds_cRain[i][1] = Math.max(0, rClouds_cRain[i][0] * 1.5 + 20 + Math.random() * 5);
        }

        Function<Double, Double> loss_w, loss_b, gradient_w, gradient_b;

        for (int i = 0; i < 10000; i++) {
            //double prediction = clouds_to_rain.apply(rClouds_cRain[0][0]);
            //System.out.println("Pred: " + prediction + " Real: " + rClouds_cRain[0][1]);
            //System.out.println("Loss: " + Math.pow(prediction - Math.max(0, rClouds_cRain[0][0] * 1.5 + 20), 2));

            for (double[] example : rClouds_cRain) {

                loss_w = (w1) -> Math.pow(w1 * example[0] + b.value - example[1], 2);
                loss_b = (b1) -> Math.pow(w.value * example[0] + b1 - example[1], 2);
                gradient_w = derivative(loss_w);
                gradient_b = derivative(loss_b);

                //System.out.println("Gradient: " + gradient_w.apply(w.value));

                w.value = w.value - leaning_rate.value * gradient_w.apply(w.value);
                b.value = b.value - leaning_rate.value * gradient_b.apply(b.value);
            }
            if (i % 1000 == 0)
                System.out.println(w.value + " " + b.value);

        }
        System.out.println(w.value + " " + b.value);

    }
}