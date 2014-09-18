package jrnnlm.test;


import jrnnlm.utils.FastMath;
import org.ejml.data.DenseMatrix64F;

import java.util.Arrays;
import java.util.Random;

public class FastMathTest {

    public static void main(String[] argv) {

        testResults();
    }

    private static void testResults() {

        double[] randomNumbers = new double[1000];
        Random rand = new Random();
        for(int i = 0; i < randomNumbers.length; ++i) {
            randomNumbers[i] = rand.nextDouble() - 0.5;
        }

        DenseMatrix64F vector1 = new DenseMatrix64F(1000, 1);
        vector1.data = randomNumbers.clone();
        FastMath.softmax(vector1);

        DenseMatrix64F vector2 = new DenseMatrix64F(1000, 1);
        vector2.data = randomNumbers.clone();
        FastMath.fastSoftmax(vector2);

        System.out.println("SOFTMAX test");
        for (int i = 0; i < 1000; ++i) {
            if (Math.abs(vector1.data[i] - vector2.data[i]) > 0.00000001) {
                System.out.print("#");
            }
        }
    }

    private static void testSpeed() {
        DenseMatrix64F vector = new DenseMatrix64F(1000, 1);

        long startTime, endTime;

        startTime = System.currentTimeMillis();
        for (int i = 0; i < 10000; ++i) {
            Arrays.fill(vector.data, 0.234);
            FastMath.sigmoid(vector);
        }
        endTime = System.currentTimeMillis();
        System.out.println(String.format("FastMath.sigmoid: %d ms", endTime - startTime));


        startTime = System.currentTimeMillis();
        for (int i = 0; i < 10000; ++i) {
            Arrays.fill(vector.data, 0.234);
            FastMath.fastSigmoid(vector);
        }
        endTime = System.currentTimeMillis();
        System.out.println(String.format("FastMath.fastSigmoid: %d ms", endTime - startTime));

        double[] randomNumbers = new double[1000];
        Random rand = new Random();
        for(int i = 0; i < randomNumbers.length; ++i) {
            randomNumbers[i] = rand.nextDouble();
        }

        startTime = System.currentTimeMillis();
        for (int i = 0; i < 10000; ++i) {
            vector.data = randomNumbers.clone();
            FastMath.softmax(vector);
        }
        endTime = System.currentTimeMillis();
        System.out.println(String.format("FastMath.softmax: %d ms", endTime - startTime));

        startTime = System.currentTimeMillis();
        for (int i = 0; i < 10000; ++i) {
            vector.data = randomNumbers.clone();
            FastMath.fastSoftmax(vector);
        }
        endTime = System.currentTimeMillis();
        System.out.println(String.format("FastMath.fastSoftmax: %d ms", endTime - startTime));

    }
}
