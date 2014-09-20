package jrnnlm.test;

import jrnnlm.core.RNNLM;
import jrnnlm.core.RNNLMConfiguration;

import java.io.IOException;

public class RNNLMComputeTest {

    public static void main(String[] argv) throws IOException {

        RNNLMConfiguration conf = new RNNLMConfiguration();
        conf.vocabSize = 2;
        conf.hiddenSize = 2;
        RNNLM lm = new RNNLM(conf);
        lm.inputSynapse.weights.setData(new double[]{30,0,0,30});
        lm.hiddenSynapse.weights.setData(new double[]{1,0,0,1});

        lm.compute(0, 1);
        lm.outputLayer.neurons.print();

        lm.compute(1, 0);
        lm.outputLayer.neurons.print();
    }
}
