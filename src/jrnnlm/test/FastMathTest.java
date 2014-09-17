package jrnnlm.test;

import jrnnlm.utils.FastMath;
import jrnnlm.utils.Logger;

public class FastMathTest {

    public static void main(String[] argv) {

        Logger.println(String.format("sigmoid: 0.5 -> %f", FastMath.sigmoid(0.5)));
        Logger.println(String.format("sigmoid: 0 -> %f", FastMath.sigmoid(0)));
        Logger.println(String.format("sigmoid: -0.5 -> %f", FastMath.sigmoid(-0.5)));
        Logger.println(String.format("sigmoid: 2 -> %f", FastMath.sigmoid(2)));

        Logger.println(String.format("typical sigmoid: 0.5 -> %f", FastMath.accurateSigmoid(0.5)));
        Logger.println(String.format("typical sigmoid: 0 -> %f", FastMath.accurateSigmoid(0)));
        Logger.println(String.format("typical sigmoid: -0.5 -> %f", FastMath.accurateSigmoid(-0.5)));
        Logger.println(String.format("typical sigmoid: 2 -> %f", FastMath.accurateSigmoid(2)));

        Logger.println("--- Time Test ---");
        double x;
        long startTime = System.nanoTime();
        for(int i = 0; i < 100000; ++i) {
            x = FastMath.sigmoid(1.2342);
        }
        long duration = System.nanoTime() - startTime;
        Logger.println(String.format("FastMath: %d ns", duration));


        startTime = System.nanoTime();
        for(int i = 0; i < 100000; ++i) {
            x = FastMath.accurateSigmoid(1.2342);
        }
        duration = System.nanoTime() - startTime;
        Logger.println(String.format("SlowMath: %d ns", duration));

    }
}
