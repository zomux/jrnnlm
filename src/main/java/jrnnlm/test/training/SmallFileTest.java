package jrnnlm.test.training;

import jrnnlm.core.RNNLM;
import jrnnlm.core.RNNLMConfiguration;

import java.io.File;
import java.io.IOException;

public class SmallFileTest {

    public static void main(String[] argv) throws IOException {

        RNNLMConfiguration conf = new RNNLMConfiguration();
        conf.trainFile = new File("data/ptb.train.100.txt");
        conf.validFile = new File("data/ptb.valid.txt");
        conf.hiddenSize = 30;
        conf.maxIters = 50;
        conf.fastMath = true;

        RNNLM lm = new RNNLM(conf);
        lm.train();
    }
}
