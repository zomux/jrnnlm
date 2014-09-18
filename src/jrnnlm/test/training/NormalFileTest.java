package jrnnlm.test.training;

import jrnnlm.core.RNNLM;
import jrnnlm.core.RNNLMConfiguration;

import java.io.File;
import java.io.IOException;

public class NormalFileTest {

    public static void main(String[] argv) throws IOException {

        RNNLMConfiguration conf = new RNNLMConfiguration();
        conf.trainFile = new File("data/ptb.train.10k.txt");
        conf.validFile = new File("data/ptb.valid.txt");
        conf.hiddenSize = 100;
        conf.maxIters = 50;
        conf.fastMath = true;

        RNNLM lm = new RNNLM(conf);
        lm.train();
    }
}
