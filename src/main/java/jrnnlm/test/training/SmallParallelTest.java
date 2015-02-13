package jrnnlm.test.training;

import java.io.File;
import java.io.FileNotFoundException;

import jrnnlm.core.RNNLMConfiguration;
import jrnnlm.io.FileInputStreamFactory;
import jrnnlm.parallel.ParallelTrainer;

public class SmallParallelTest {

    public static void main(String[] argv) throws FileNotFoundException {

        RNNLMConfiguration conf = new RNNLMConfiguration();
        conf.trainStreamFactory = new FileInputStreamFactory(new File("data/ptb.train.100.txt"));
        conf.validFile = new File("data/ptb.valid.txt");
        conf.hiddenSize = 30;
        conf.maxIters = 50;
        conf.fastMath = true;

        ParallelTrainer trainer = new ParallelTrainer(conf, 4);
        trainer.train();


    }
}
