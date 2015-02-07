package jrnnlm.core;

import java.io.IOException;

public class LMSaver {

    private Layer inputLayer;
    private Layer outputLayer;
    private Layer hiddenLayer;
    private Layer recurrentLayer;
    private Synapse inputSynapse;
    private Synapse recurrentSynapse;
    private Synapse hiddenSynapse;


    public void save(RNNLM lm) {

        inputLayer = lm.inputLayer.copy();
        outputLayer = lm.outputLayer.copy();
        hiddenLayer = lm.hiddenLayer.copy();
        recurrentLayer = lm.recurrentLayer.copy();

        inputSynapse = lm.inputSynapse.copy();
        recurrentSynapse = lm.recurrentSynapse.copy();
        hiddenSynapse = lm.hiddenSynapse.copy();
    }

    public void restore(RNNLM lm) {

        lm.inputLayer = inputLayer.copy();
        lm.outputLayer = outputLayer.copy();
        lm.hiddenLayer = hiddenLayer.copy();
        lm.recurrentLayer = recurrentLayer.copy();

        lm.inputSynapse = inputSynapse.copy();
        lm.recurrentSynapse = recurrentSynapse.copy();
        lm.hiddenSynapse = hiddenSynapse.copy();
    }

    public void zero(RNNLMConfiguration conf) {

        RNNLM lm = null;
        try {
            lm = new RNNLM(conf);
        } catch (IOException e) {
            e.printStackTrace();
        }

        if (lm != null) {
            lm.inputSynapse.zero();
            lm.recurrentSynapse.zero();
            lm.hiddenSynapse.zero();
            save(lm);
        }

    }

    public void mergeSynapse(RNNLM lm, double alpha) {

        mergeArray(inputSynapse.weights.data, lm.inputSynapse.weights.data, alpha);
        mergeArray(recurrentSynapse.weights.data, lm.recurrentSynapse.weights.data, alpha);
        mergeArray(hiddenSynapse.weights.data, lm.hiddenSynapse.weights.data, alpha);
    }

    private void mergeArray(double[] targetArray, double[] sourceArray, double alpha) {

        for (int i = 0; i < targetArray.length; ++i) {
            targetArray[i] += alpha * sourceArray[i];
        }
    }
}
