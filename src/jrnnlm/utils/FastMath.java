package jrnnlm.utils;

import org.ejml.data.DenseMatrix64F;

public class FastMath {

    public static double approxExp(double val) {

        final long tmp = (long) (1512775 * val + 1072632447);
        return Double.longBitsToDouble(tmp << 32);
    }

    public static void fastSigmoid(DenseMatrix64F vector) {

        double[] data = vector.getData();
        for (int i = 0; i < data.length; ++i) {
            if (data[i] > 50) data[i] = 50;
            if (data[i] < -50) data[i] = -50;
            data[i] = -data[i];
        }
        info.yeppp.Math.Exp_V64f_V64f(data, 0, data, 0, data.length);
        for (int i = 0; i < data.length; ++i) {
            data[i] = 1 / (1 + data[i]);
        }
    }

    public static void sigmoid(DenseMatrix64F vector) {

        double[] data = vector.getData();
        double val;
        for (int i = 0; i < data.length; ++i) {
            val = data[i];
            if (val > 50) val = 50;
            if (val < -50) val = -50;
            data[i] = 1 / (1 + Math.exp(-val));
        }
    }


    public static void fastSoftmax(DenseMatrix64F vector) {

        double[] data = vector.getData();
        double maxAc = info.yeppp.Core.Max_V64f_S64f(data, 0, data.length);
        info.yeppp.Core.Subtract_IV64fS64f_IV64f(data, 0, maxAc, data.length);
        info.yeppp.Math.Exp_V64f_V64f(data, 0, data, 0, data.length);
        double sum = info.yeppp.Core.Sum_V64f_S64f(data, 0, data.length);
        double sumDivided = 1.0 / sum;
        info.yeppp.Core.Multiply_IV64fS64f_IV64f(data, 0, sumDivided, data.length);
    }

    public static void softmax(DenseMatrix64F vector) {

        double[] data = vector.getData();
        double sum = 0;
        double maxAc = - Float.MAX_VALUE;
        for (int i = 0; i < data.length; ++i) {
            if (data[i] > maxAc) maxAc =data[i];
        }
        for (int i = 0; i < data.length; ++i) {
            data[i] = Math.exp(data[i] - maxAc);
            sum += data[i];
        }
        for (int i = 0; i < data.length; ++i) {
            data[i] = data[i] / sum;
        }
    }


}
