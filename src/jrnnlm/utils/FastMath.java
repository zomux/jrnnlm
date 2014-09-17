package jrnnlm.utils;

import org.ejml.data.DenseMatrix64F;

public class FastMath {

    public static double exp(double val) {

        return Math.exp(val);
        //final long tmp = (long) (1512775 * val + 1072632447);
        //return Double.longBitsToDouble(tmp << 32);
    }

    public static double sigmoid(double x) {

        return 1 / (1 + FastMath.exp(-x));
    }

    public static void sigmoid(DenseMatrix64F vector) {

        double[] data = vector.getData();
        double val;
        for (int i = 0; i < data.length; ++i) {
            val = data[i];
            if (val > 50) val = 50;
            if (val < -50) val = -50;
            data[i] = sigmoid(val);
        }
    }

    public static void softmax(DenseMatrix64F vector) {

        double[] data = vector.getData();
        double sum = 0;
        double maxAc = - Float.MAX_VALUE;
        for (int i = 0; i < data.length; ++i) {
            if (data[i] > maxAc) maxAc =data[i];
        }
        for (int i = 0; i < data.length; ++i) {
            data[i] = FastMath.exp(data[i] - maxAc);
            sum += data[i];
        }
        for (int i = 0; i < data.length; ++i) {
            data[i] = data[i] / sum;
        }
    }

    public static double accurateSigmoid(double x) {

        return 1 / (1 + Math.exp(-x));
    }


}
