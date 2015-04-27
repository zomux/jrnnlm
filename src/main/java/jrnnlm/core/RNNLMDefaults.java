package jrnnlm.core;

public class RNNLMDefaults {

    public static final int MAX_NGRAM_ORDER = 20;

    public static final int MAX_ITERS = 10;

    public static final double STARTING_ALPHA = 0.1;

    public static final double REGULARIZATION = 0.0000001;

    public static final double DYNAMIC = 0;

    public static final double MIN_IMPROVEMENT = 1.003;

    public static final int BPTT = 3;

    public static final int BPTT_BLOCK = 10;

    public static final int HIDDEN_SIZE = 30;

    public static final int VOCAB_SIZE = 1000;

    public static final int CLASS_SIZE = 0;

    public static final boolean STRENGTHEN_LAST_WORD = false;
    public static final int DIRECT_ORDER = 3;

    public static final boolean INDEPENDENT = false;

    public static final boolean ALPHA_DIVIDE = false;

    public static final boolean ALWAYS_BPTT = true;

    public static final boolean SAME_INITIAL_WEIGHTS = false;

    public static final boolean FAST_MATH = true;

    public static final int RANDOM_SEED = 3;
}
