package jrnnlm.test.training;

import java.io.File;
import java.io.IOException;

import jrnnlm.core.RNNLM;
import jrnnlm.core.RNNLMConfiguration;
import jrnnlm.io.FileInputStreamFactory;

public class SmallFileTest {

    public static void main(String[] argv) throws IOException {
        RNNLMConfiguration conf = new RNNLMConfiguration();
        conf.trainStreamFactory = new FileInputStreamFactory(new File("data/ptb.train.100.txt"));
        conf.validFile = new File("data/ptb.valid.txt");
        conf.hiddenSize = 30;
        conf.maxIters = 50;
        conf.fastMath = true;

        RNNLM lm = new RNNLM(conf);
        lm.train();
    }
}
