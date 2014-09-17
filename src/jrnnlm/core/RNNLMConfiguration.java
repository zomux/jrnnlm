package jrnnlm.core;

import java.io.File;

public class RNNLMConfiguration {

    // Learning rate
    public double startingAlpha = RNNLMDefaults.STARTING_ALPHA;
    // Regularization
    public double regularization = RNNLMDefaults.REGULARIZATION;

    public int vocabSize = RNNLMDefaults.VOCAB_SIZE;
    public int hiddenSize = RNNLMDefaults.HIDDEN_SIZE;
    public int classSize = RNNLMDefaults.CLASS_SIZE;

    public int maxNgramOrder = RNNLMDefaults.MAX_NGRAM_ORDER;
    public int maxIters = RNNLMDefaults.MAX_ITERS;

    public int bptt = RNNLMDefaults.BPTT;
    public int bpttBlock = RNNLMDefaults.BPTT_BLOCK;

    public double minImprovement = RNNLMDefaults.MIN_IMPROVEMENT;
    public double reguralization = RNNLMDefaults.REGULARIZATION;
    public double dynamic = RNNLMDefaults.DYNAMIC;

    public boolean strengthenLastWord = RNNLMDefaults.STRENGTHEN_LAST_WORD;

    public int directOrder = RNNLMDefaults.DIRECT_ORDER;
    public boolean independent = RNNLMDefaults.INDEPENDENT;
    public boolean alphaDivide = RNNLMDefaults.ALPHA_DIVIDE;

    public boolean alwaysBPTT = RNNLMDefaults.ALWAYS_BPTT;

    // Files
    public File trainFile = null;
    public int[] trainData = null;

}
