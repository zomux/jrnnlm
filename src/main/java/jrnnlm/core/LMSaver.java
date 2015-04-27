package jrnnlm.core;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;

public class LMSaver implements Serializable {

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
    
    public static void save(RNNLM lm, String outputDir) throws FileNotFoundException, IOException{
      LMSaver saver = new LMSaver();
      saver.save(lm);
      ObjectOutputStream oos = null;
      
      // write configuration file
      oos = new ObjectOutputStream(new FileOutputStream(outputDir + "/conf.obj"));
      oos.writeObject(lm.getConfiguration());
      oos.close();
      // write weights
      oos = new ObjectOutputStream(new FileOutputStream(outputDir + "/saver.obj"));
      oos.writeObject(saver);
      oos.close();
    }

    public static RNNLM load(String inputDir) throws FileNotFoundException, IOException, ClassNotFoundException {
      ObjectInputStream ois = new ObjectInputStream(new FileInputStream(inputDir + "/conf.obj"));
      RNNLMConfiguration conf = (RNNLMConfiguration) ois.readObject();
      ois.close();
      
      RNNLM rnnlm = new RNNLM(conf);
      
      ois = new ObjectInputStream(new FileInputStream(inputDir + "/saver.obj"));
      LMSaver saver = (LMSaver) ois.readObject();
      ois.close();
      saver.restore(rnnlm);
      
      return rnnlm;
      
    }
}
