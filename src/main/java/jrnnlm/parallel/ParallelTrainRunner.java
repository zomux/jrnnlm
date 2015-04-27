package jrnnlm.parallel;

import jrnnlm.core.LMSaver;
import jrnnlm.core.RNNLM;
import jrnnlm.core.RNNLMConfiguration;
import jrnnlm.utils.Logger;

import java.io.IOException;


public class ParallelTrainRunner implements Runnable {

    public RNNLM lm;
    public double finalEntropy = 0;

    public ParallelTrainRunner(RNNLMConfiguration conf, LMSaver saver) {

        lm = null;
        try {
            lm = new RNNLM(conf);
        } catch (IOException e) {
            Logger.error("[ParallelTrainRunner] error");
            e.printStackTrace();
        }

        if (saver != null && lm != null) {
            saver.restore(lm);
        }
    }

    @Override
    public void run() {

        if (lm != null) {
            finalEntropy = lm.train();
        }
    }
}
