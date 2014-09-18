package jrnnlm.core;

import org.ejml.data.DenseMatrix64F;

import java.util.Arrays;

public class Layer {

    public DenseMatrix64F activations;
    public DenseMatrix64F errors;
    public final int size;

    public Layer(int size) {

        this.size = size;
        activations = new DenseMatrix64F(size, 1);
        errors = new DenseMatrix64F(size, 1);
        zero();
    }

    public void zero() {

        activations.zero();
        errors.zero();
    }

    public void fillNeuronsByOne() {

        Arrays.fill(activations.getData(), 1);
    }

    public void copyFrom(Layer layer) {

        activations = layer.activations.copy();
        errors = layer.errors.copy();
    }

    public void errorDerivation() {

        double[] neuronData = activations.getData();
        double[] errorData = errors.getData();

        for(int i = 0; i < size; ++i) {
            errorData[i] = errorData[i] * neuronData[i] * (1 - neuronData[i]);
        }
    }

    public Layer copy() {

        Layer layer = new Layer(size);
        layer.activations = activations.copy();
        layer.errors = errors.copy();
        return layer;
    }

}
