package jrnnlm.test;

import org.ejml.alg.dense.mult.MatrixVectorMult;
import org.ejml.data.DenseMatrix64F;
import org.jblas.DoubleMatrix;

import java.util.Random;

public class BLASTest {

    public static void main(String[] argv) {

        speedComaprsion();
    }

    private static void speedComaprsion() {

        Random rand = new Random();
        double[][] originalData = new double[100][200];
        for (int i = 0; i < 100; ++i) {
            double[] row = new double[200];
            for (int j = 0; j < 200; ++j) {
                row[j] = rand.nextDouble();
            }
            originalData[i] = row;
        }

        double[] originalVector = new double[200];
        for (int j = 0; j < 200; ++j) {
            originalVector[j] = rand.nextDouble();
        }

        DoubleMatrix m1 = new DoubleMatrix(originalData.clone());
        DenseMatrix64F m2 = new DenseMatrix64F(originalData.clone());

        DoubleMatrix v1 = new DoubleMatrix(originalVector.clone());
        DenseMatrix64F v2 = new DenseMatrix64F(200, 1);
        DenseMatrix64F v3 = new DenseMatrix64F(100, 1);
        v2.data = originalVector.clone();

        m1.mmul(v1);
        MatrixVectorMult.mult(m2, v2, v3);
        System.out.println("SAFD");
    }
}
