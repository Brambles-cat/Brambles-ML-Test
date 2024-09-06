package com.brambles.datatypes;

public class GrayScaleImage extends Example {
    public int label;
    public double[] image;

    public transient double[] labels = new double[10];

    @Override
    public void postDeserialization() {
        labels[label] = 1d;
    }

    @Override
    public double[] getLabels() {
        return labels;
    }

    @Override
    public double[] getFeatures() {
        return image;
    }
}
