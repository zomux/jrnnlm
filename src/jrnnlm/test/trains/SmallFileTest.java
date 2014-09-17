package jrnnlm.test.trains;

import jrnnlm.core.RNNLM;
import jrnnlm.core.RNNLMConfiguration;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;

public class SmallFileTest {

    public static void main(String[] argv) throws IOException {

        RNNLMConfiguration conf = new RNNLMConfiguration();
        conf.trainFile = new File("data/ptb.train.100.txt");
        conf.hiddenSize = 30;
        conf.maxIters = 20;

        RNNLM lm = new RNNLM(conf);

        Arrays.fill(lm.inputSynapse.weights.getData(), 0.1);
        Arrays.fill(lm.recurrentSynapse.weights.getData(), 0.1);
        Arrays.fill(lm.hiddenSynapse.weights.getData(), 0.1);
        lm.train();
    }
}
