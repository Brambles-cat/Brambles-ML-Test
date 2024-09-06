package com.brambles.datatypes;

public abstract class Example {
    public abstract double[] getLabels();
    public abstract double[] getFeatures();

    public void postDeserialization() {}
}
