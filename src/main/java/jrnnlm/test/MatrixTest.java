package jrnnlm.test;

import org.ejml.data.DenseMatrix64F;

import java.util.Arrays;

public class MatrixTest {

    public static void main(String argv[]) {

        DenseMatrix64F m = new DenseMatrix64F(2, 2);
        m.setData(new double[]{1,2,3,4});
        m.print();
        System.out.println(Arrays.toString(m.getData()));

        m = new DenseMatrix64F(2, 1);
        m.setData(new double[]{1,2});
        m.print();
        System.out.println(Arrays.toString(m.getData()));

        // In-place modification
        m = new DenseMatrix64F(2, 2);
        m.setData(new double[]{1,2,3,4});
        double[] data = m.getData();
        data[0] = 10;
        m.print();
        System.out.println(Arrays.toString(m.getData()));
    }
}
