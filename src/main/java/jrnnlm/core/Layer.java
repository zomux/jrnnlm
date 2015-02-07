package jrnnlm.core;

import org.ejml.data.DenseMatrix64F;

import java.util.Arrays;

public class Layer {

    public DenseMatrix64F neurons;
    public DenseMatrix64F errors;
    public final int size;

    public Layer(int size) {

        this.size = size;
        neurons = new DenseMatrix64F(size, 1);
        errors = new DenseMatrix64F(size, 1);
        zero();
    }

    public void zero() {

        neurons.zero();
        errors.zero();
    }

    public void fillNeuronsByOne() {

        Arrays.fill(neurons.getData(), 1);
    }

    public void copyFrom(Layer layer) {

        neurons = layer.neurons.copy();
        errors = layer.errors.copy();
    }

    public void errorDerivation() {

        double[] neuronData = neurons.getData();
        double[] errorData = errors.getData();

        for(int i = 0; i < size; ++i) {
            errorData[i] = errorData[i] * neuronData[i] * (1 - neuronData[i]);
        }
    }

    public Layer copy() {

        Layer layer = new Layer(size);
        layer.neurons = neurons.copy();
        layer.errors = errors.copy();
        return layer;
    }

}
