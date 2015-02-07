package jrnnlm.test.training;

import jrnnlm.core.RNNLMConfiguration;
import jrnnlm.parallel.ParallelTrainer;

import java.io.File;

public class SmallParallelTest {

    public static void main(String[] argv) {

        RNNLMConfiguration conf = new RNNLMConfiguration();
        conf.trainFile = new File("data/ptb.train.100.txt");
        conf.validFile = new File("data/ptb.valid.txt");
        conf.hiddenSize = 30;
        conf.maxIters = 50;
        conf.fastMath = true;

        ParallelTrainer trainer = new ParallelTrainer(conf, 4);
        trainer.train();


    }
}
