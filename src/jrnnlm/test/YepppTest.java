package jrnnlm.test;

import java.util.Arrays;

public class YepppTest {

    public static void main(String[] argv) {

        // Test for exp
        double[] testcase = new double[100];
        Arrays.fill(testcase, 0.234);

        info.yeppp.Math.Exp_V64f_V64f(testcase, 0, testcase, 0, testcase.length);
        System.out.println(Arrays.toString(testcase));

        testcase = new double[]{-0.1, -0.2, 0.0};
        System.out.println(info.yeppp.Core.Sum_V64f_S64f(testcase, 0, testcase.length));


    }
}
