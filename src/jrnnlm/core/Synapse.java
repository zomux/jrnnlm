package jrnnlm.core;

import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;

import java.util.Random;

public class Synapse {

    public DenseMatrix64F weights;

    public Synapse(int size1, int size2) {

        weights = new DenseMatrix64F(size2, size1);
        randomlyInit();
    }

    private void randomlyInit() {

        Random r = new Random(RNNLMDefaults.RANDOM_SEED);
        for (int i = 0; i < weights.numRows; ++i) {
            for (int j = 0; j < weights.numCols; ++j) {
                double value = (r.nextDouble() + r.nextDouble() + r.nextDouble()) - 1.5;
                weights.set(i, j, value);
            }
        }
    }

    public void zero() {

        weights.zero();
    }

    public void learnOneColumn(int col, DenseMatrix64F verticalVector, double alpha) {

        if (col < 0) return;
        double[] data = weights.getData();
        double[] vectorData = verticalVector.getData();
        int rowCount = weights.numRows;
        int colCount = weights.numCols;

        for (int row = 0; row < rowCount; ++row) {
            data[col + row * colCount] += alpha * vectorData[row];
        }
    }

    public void learnAll(DenseMatrix64F horizontalVector, DenseMatrix64F verticalVector, double alpha, double beta) {

        double[] horizontalData = horizontalVector.getData();
        double[] verticalData = verticalVector.getData();
        double[] data = weights.getData();
        int rowCount = weights.numRows;
        int colCount = weights.numCols;

        for (int row = 0; row < rowCount; ++row) {
            for (int col = 0; col < colCount; ++col) {
                if (beta == Double.NEGATIVE_INFINITY) {
                    data[col + row * colCount] += alpha * horizontalData[col] * verticalData[row];
                }
                else {
                    data[col + row * colCount] += alpha * horizontalData[col] * verticalData[row] - beta * data[col + row * colCount];
                }
            }
        }
    }

    public void accumulate(DenseMatrix64F anotherMatrix, double beta) {

        double[] data = weights.getData();
        double[] anotherData = anotherMatrix.getData();
        int rowCount = weights.numRows;
        int colCount = weights.numCols;
        double oneMinusBeta = (1 - beta);

        for (int row = 0; row < rowCount; ++row) {
            for (int col = 0; col < colCount; ++col) {
                data[col + row * colCount] *= oneMinusBeta;
                data[col + row * colCount] += anotherData[col + row * colCount];
            }
        }
    }

    public Synapse copy() {

        Synapse synapse = new Synapse(weights.numCols, weights.numRows);
        synapse.weights = weights.copy();
        return synapse;
    }
}
