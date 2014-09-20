package jrnnlm.core;

import javafx.util.Pair;
import jrnnlm.core.scanner.FileScanner;
import jrnnlm.core.scanner.RawScanner;
import jrnnlm.core.scanner.WordIndexScanner;
import jrnnlm.utils.FastMath;
import jrnnlm.utils.Logger;
import org.ejml.alg.dense.mult.MatrixVectorMult;
import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;

import java.io.IOException;
import java.util.Arrays;

public class RNNLM {

    // Sizes
    private RNNLMConfiguration conf;

    // Layers
    public Layer inputLayer;
    public Layer recurrentLayer;
    public Layer hiddenLayer;
    public Layer outputLayer;

    public Synapse inputSynapse;
    public Synapse hiddenSynapse;
    public Synapse recurrentSynapse;

    public DenseMatrix64F wordVector;

    // Learning rates
    private double alpha;
    private double startingAlpha;
    private double beta;
    private double beta2;

    private int counter = 0;

    // BPTT
    private int[] bpttHistory;
    private int[] history;
    private Layer[] bpttHiddenLayers;
    private Synapse bpttRecurrentSynapse;
    private Synapse bpttInputSynapse;

    // Vocab
    private Vocabulary vocab = null;

    private LMSaver saver;

    public RNNLM(RNNLMConfiguration conf) throws IOException {

        this.conf = conf;

        // For files
        if (conf.vocab != null) {
            vocab = conf.vocab;
            conf.vocabSize = vocab.size();
        }
        else if (conf.trainFile != null) {
            vocab = new Vocabulary();
            vocab.loadRawText(conf.trainFile);
            conf.vocabSize = vocab.size();
        }

        inputLayer = new Layer(inputLayerSize());
        recurrentLayer = new Layer(hiddenLayerSize());
        hiddenLayer = new Layer(hiddenLayerSize());
        outputLayer = new Layer(outputLayerSize());

        inputSynapse = new Synapse(inputLayerSize(), hiddenLayerSize());
        hiddenSynapse = new Synapse(hiddenLayerSize(), outputLayerSize());
        recurrentSynapse = new Synapse(hiddenLayerSize(), hiddenLayerSize());

        bpttHiddenLayers = new Layer[(conf.bptt+conf.bpttBlock+1)];
        for (int i = 0; i < bpttHiddenLayers.length; ++i) {
            bpttHiddenLayers[i] = new Layer(hiddenLayerSize());
        }

        bpttInputSynapse = new Synapse(inputLayer.size, hiddenLayer.size);
        bpttRecurrentSynapse = new Synapse(recurrentLayer.size, hiddenLayer.size);
        bpttInputSynapse.zero();
        bpttRecurrentSynapse.zero();

        history = new int[conf.maxNgramOrder];
        bpttHistory = new int[conf.bptt + conf.bpttBlock + 1];

        wordVector = new DenseMatrix64F(outputLayerSize(), 1);
        saver = new LMSaver();

        // Weights
        if (conf.sameInitialWeights) {
            Arrays.fill(inputSynapse.weights.getData(), 0.1);
            Arrays.fill(recurrentSynapse.weights.getData(), 0.1);
            Arrays.fill(hiddenSynapse.weights.getData(), 0.1);
        }
    }

    public int inputLayerSize() {

        return conf.vocabSize;
    }

    public int hiddenLayerSize() {

        return conf.hiddenSize;
    }

    public int outputLayerSize() {

        return conf.vocabSize + conf.classSize;
    }

    public void reset() {

        hiddenLayer.fillNeuronsByOne();
        recurrentLayer.fillNeuronsByOne();
        Arrays.fill(bpttHistory, 0);

        for (int i = 2; i < conf.bptt + conf.bpttBlock; ++i) {
            bpttHiddenLayers[i].zero();
        }
    }

    public void flush() {

        inputLayer.zero();
        hiddenLayer.zero();
        outputLayer.zero();
        Arrays.fill(recurrentLayer.neurons.data, 0.1);
        recurrentLayer.errors.zero();
    }

    private void initParameters() {

        alpha = startingAlpha = conf.startingAlpha;
        beta = conf.regularization;
    }

    private void clearHistory() {

        Arrays.fill(history, 0);
        Arrays.fill(bpttHistory, -1);
    }

    public double train() {

        // Load word scanner
        WordIndexScanner scanner = null;
        if (conf.trainFile == null && conf.trainData == null) {
            Logger.error("trainFile or trainData must be set");
        }
        if (conf.trainFile != null) {
            try {
                scanner = new FileScanner(vocab, conf.trainFile);
            }
            catch (IOException e) {
                e.printStackTrace();
            }
        }
        else {
            scanner = new RawScanner(conf.trainData);
        }

        // Valid file scanner
        WordIndexScanner validScanner = null;
        if (conf.validFile == null && conf.validData == null) {
            Logger.error("validFile or validData must be set");
        }
        if (conf.trainFile != null) {
            try {
                validScanner = new FileScanner(vocab, conf.validFile);
            }
            catch (IOException e) {
                e.printStackTrace();
            }
        }
        else {
            validScanner = new RawScanner(conf.validData);
        }

        int lastWord, word;
        double logp = 0;
        double lastlogp = -10000000;
        double entroy = 0;
        initParameters();

        long startTime = System.currentTimeMillis();

        for (int iterNumber = 0; iterNumber < conf.maxIters; ++iterNumber) {

            clearHistory();
            flush();
            lastWord = 0;

            while((word = scanner.next()) != -1) {
                ++counter;
                if (counter % 10000 == 0) System.out.print("#");
                compute(lastWord, word);
                logp += Math.log10(outputLayer.neurons.get(word, 0));

                if (Double.isInfinite(logp)) {
                    System.err.println(String.format("Infinite error: %d -> %f", word, outputLayer.neurons.get(word, 0)));
                }

                // Shift memory needed for bptt to next time step
                if (conf.bptt>0) {
                    for (int i= conf.bptt + conf.bpttBlock - 1; i > 0; --i) {
                        bpttHistory[i] = bpttHistory[i-1];
                    }
                    bpttHistory[0] = lastWord;

                    for (int i = conf.bptt + conf.bpttBlock - 1; i > 0; --i) {
                        bpttHiddenLayers[i] = bpttHiddenLayers[i - 1];
                    }
                    bpttHiddenLayers[0] = new Layer(bpttHiddenLayers[0].size);
                }

                // Back propagate
                learn(lastWord, word);

                // Copy hidden layer to recurrent layer
                recurrentLayer.neurons = hiddenLayer.neurons.copy();

                lastWord = word;
                for (int i = conf.maxNgramOrder - 1; i > 0; --i) {
                    history[i] = history[i - 1];
                }
                history[0]=lastWord;

                if (conf.independent && (word==0)) {
                    reset();
                }
            }
            System.out.print(String.format("%c", 13));

            // Info
            long endTime = System.currentTimeMillis();
            double wordsPerSecond = counter / ( (double) (endTime - startTime) / 1000);
            Logger.info(String.format("Iter: %d, Alpha %f, TRAIN entropy: %.4f, Words/sec: %.1f", iterNumber, alpha, -logp / Math.log10(2) / (counter + 1), wordsPerSecond));

            // Valid
            Pair<Double, Integer> entropyCountPair = estimate(validScanner);
            logp = entropyCountPair.getKey();

            Logger.info(String.format("Iter: %d, VALID entropy: %.4f, PPL: %.4f", iterNumber, -logp/Math.log10(2)/entropyCountPair.getValue(), Math.log10(-logp/entropyCountPair.getValue())));


            // Ending
            if (logp * conf.minImprovement < lastlogp) {
                saver.restore(this);
                if (!conf.alphaDivide) {
                    conf.alphaDivide = true;
                }
                else {
                    // Exit training
                    entroy = -lastlogp/Math.log10(2)/entropyCountPair.getValue();
                    break;
                }

            }
            else {
                saver.save(this);
            }

            if (conf.alphaDivide) alpha /= 2;
            counter = 0;
            lastlogp = logp;
            logp=0;
            scanner.reset();
            validScanner.reset();
        }
        return entroy;
    }

    public Pair<Double, Integer> estimate(WordIndexScanner scanner) {

        int lastWord=0;
        double logp=0;
        int wordCount=0;
        int word;

        flush();

        while ((word = scanner.next()) != -1) {
            // Compute probability
            compute(lastWord, word);
            logp += Math.log10(outputLayer.neurons.get(word, 0));
            wordCount += 1;

            // Copy hidden layer to recurrent layer
            recurrentLayer.neurons = hiddenLayer.neurons.copy();

            // Ending
            lastWord = word;
            for (int i = conf.maxNgramOrder - 1; i > 0; --i) {
                history[i] = history[i - 1];
            }
            history[0]=lastWord;

            if (conf.independent && (word==0)) {
                reset();
            }
        }
        return new Pair<Double, Integer>(logp, wordCount);
    }

    public void learn(int lastWord, int word) {

        if (word==-1) return;
        int bptt = conf.bptt;
        int bpttBlock = conf.bpttBlock;

        beta2=beta*alpha;

        // Compute error vectors
        loadWordVector(word);
        CommonOps.sub(wordVector, outputLayer.neurons, outputLayer.errors);

        // Flush error
        hiddenLayer.errors.zero();
        // TODO: class compression

        // TODO: learn direct connections between words

        // TODO: learn direct connections to classes

        // TODO: back propagate to compression

        CommonOps.multTransA(hiddenSynapse.weights, outputLayer.errors, hiddenLayer.errors);

        // Learn hidden weights
        hiddenSynapse.learnAll(hiddenLayer.neurons, outputLayer.errors, alpha, beta);

        // BPTT

        // Copy from HL to BL_0
        bpttHiddenLayers[0].copyFrom(hiddenLayer);

        int wordOfStep;
        int maxSteps = bptt + bpttBlock;

        if (((counter % conf.bpttBlock)==0) || (conf.independent && (word==0))) {

            for (int step = 0; step < maxSteps - 2; step++) {

                wordOfStep = bpttHistory[step];

                // Error derivation at hidden layer
                hiddenLayer.errorDerivation();

                // IBS[col of word_step][x] += alpha * HL.er[x] * 1
                bpttInputSynapse.learnOneColumn(wordOfStep, hiddenLayer.errors, alpha);

                // RBS[x] += alpha * HL.er[row of x] * RL.ac[col of x]
                bpttRecurrentSynapse.learnAll(recurrentLayer.neurons, hiddenLayer.errors, alpha, Double.NEGATIVE_INFINITY);

                // Clean RL.er
                recurrentLayer.errors.zero();

                // Propagate HL.er -> (RS) -> RL.er
                CommonOps.multTransA(recurrentSynapse.weights, hiddenLayer.errors, recurrentLayer.errors);

                // HL.er = RL.er + BL_s+1.er
                CommonOps.add(bpttHiddenLayers[step + 1].errors, recurrentLayer.errors, hiddenLayer.errors);

                if (step < maxSteps - 3) {
                    hiddenLayer.neurons = bpttHiddenLayers[step + 1].neurons.copy();
                    recurrentLayer.neurons = bpttHiddenLayers[step + 2].neurons.copy();
                }
            }

            // Clean BL_x.er
            for (Layer bpttLayer : bpttHiddenLayers) {
                bpttLayer.errors.zero();
            }

            // HL.ac = BL_0.ac
            hiddenLayer.neurons = bpttHiddenLayers[0].neurons.copy();

            // RS += RBS - regularization (self * beta2)
            recurrentSynapse.accumulate(bpttRecurrentSynapse.weights, (counter%10)==0 ? beta2 : 0);

            // Clean RBS
            bpttRecurrentSynapse.zero();

            // IS += IBS - regularization (self * beta2)
            inputSynapse.accumulate(bpttInputSynapse.weights, (counter%10)==0 ? beta2 : 0);

            // Clean IBS
            bpttInputSynapse.zero();
        }
    }

    public void compute(int lastWord, int word) {

        double activeValue = conf.strengthenLastWord ? 2 : 1;
        if (lastWord != -1) inputLayer.neurons.set(lastWord, 0, activeValue);

        //Propagate input -> hidden
        MatrixVectorMult.mult(inputSynapse.weights, inputLayer.neurons, hiddenLayer.neurons);
        MatrixVectorMult.multAdd(recurrentSynapse.weights, recurrentLayer.neurons, hiddenLayer.neurons);

        // hidden -> (sigmoid) -> hidden
        if (conf.fastMath) {
            FastMath.fastSigmoid(hiddenLayer.neurons);
        }
        else {
            FastMath.sigmoid(hiddenLayer.neurons);
        }

        // TODO: Add compression layer

        // TODO: Support class

        //Propagate hidden -> output
        outputLayer.neurons.zero();
        MatrixVectorMult.mult(hiddenSynapse.weights, hiddenLayer.neurons, outputLayer.neurons);

        // TODO: apply direct connections to classes

        // TODO: apply direct connections to words

        // output -> (softmax) -> output

        if (word!=-1) {
            if (conf.classSize == 0) {
                if (conf.fastMath) {
                    // TODO: fastSoftmax make convergence slow here, unveiled by SmallFileTest
                    FastMath.fastSoftmax(outputLayer.neurons);
                }
                else {
                    FastMath.softmax(outputLayer.neurons);
                }

            } else {
                // TODO: support class
                System.err.println("no class support now!");
            }
        }

        if (lastWord != -1) inputLayer.neurons.set(lastWord, 0, 0);


    }

    private void loadWordVector(int word) {

        wordVector.zero();
        wordVector.set(word, 0, 1);
    }
}
