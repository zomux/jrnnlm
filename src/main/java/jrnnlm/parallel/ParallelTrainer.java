package jrnnlm.parallel;

import javafx.util.Pair;
import jrnnlm.core.LMSaver;
import jrnnlm.core.RNNLM;
import jrnnlm.core.RNNLMConfiguration;
import jrnnlm.core.Vocabulary;
import jrnnlm.core.scanner.FileScanner;
import jrnnlm.utils.Logger;

import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

public class ParallelTrainer {

    private static final File SPLIT_PATH = new File("/tmp");
    private final int cores;
    private final RNNLMConfiguration conf;

    public ParallelTrainer(RNNLMConfiguration conf, int cores) {

        this.cores = cores;
        this.conf = conf;
    }

    public void train() {

        // Build vocab
        Vocabulary vocab = new Vocabulary();
        try {
            vocab.loadRawText(conf.trainFile);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            return;
        }
        conf.vocab = vocab;

        // Train
        LMSaver saver = null;
        double entropy = 1000000;
        while (true) {
            Pair<Double, LMSaver> scoreSaverPair = trainAndMerge(saver);
            double averageEntropy = scoreSaverPair.getKey();
            if (averageEntropy > entropy) break;
            entropy = averageEntropy;
            Logger.info(String.format("[Overall] average entropy: %.4f", averageEntropy));
            saver = scoreSaverPair.getValue();
        }

        // Final Valid
        double finalEntropy = 0;
        try {
            RNNLM lm = new RNNLM(conf);
            if (lm != null && saver != null) {
                saver.restore(lm);
                Pair<Double, Integer> logpPair = lm.estimate(new FileScanner(vocab, conf.validFile));
                finalEntropy = -logpPair.getKey()/Math.log10(2)/logpPair.getValue();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        if (finalEntropy != 0) {
            Logger.info(String.format("Final Valid Entropy: %.4f", finalEntropy));
        }
        else {
            Logger.info(String.format("Final Valid failed"));
        }

    }

    public Pair<Double, LMSaver> trainAndMerge(LMSaver saver) {

        List<File> splittedFiles = new ArrayList<File>();
        try {
            splittedFiles = split();
        } catch (IOException e) {
            e.printStackTrace();
        }
        Logger.info(String.format("[ParallelTrainer] split %s \n-> %s", conf.trainFile.toString(), splittedFiles.toString()));

        ExecutorService ex = Executors.newFixedThreadPool(cores);
        List<ParallelTrainRunner> runners = new ArrayList<ParallelTrainRunner>();
        for (File part: splittedFiles) {
            RNNLMConfiguration partConf = conf.clone();
            partConf.trainFile = part;
            ParallelTrainRunner runner = new ParallelTrainRunner(partConf, saver);
            runners.add(runner);
            ex.execute(runner);
        }

        ex.shutdown();
        try {
            ex.awaitTermination(Long.MAX_VALUE, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            Logger.info("[ParallelTrainer] Interrupted");
        }

        Logger.info("[ParallelTrainer] Finalize");

        // Merge
        StringBuilder scoreBuilder = new StringBuilder("[ParallelTrainer] scores: ");
        double entropySum = 0;
        for (int i = 0; i < runners.size(); ++i) {
            double finalEntropy = runners.get(i).finalEntropy;
            scoreBuilder.append(String.format("%d:%.4f ", i + 1, finalEntropy));
            entropySum += finalEntropy;
        }
        Logger.info(scoreBuilder.toString());

        LMSaver outputSaver = new LMSaver();
        outputSaver.zero(conf);
        for (int i = 0; i < runners.size(); ++i) {
            // double alpha = runners.get(i).finalEntropy / entropySum;
            double alpha = 1 / cores;
            outputSaver.mergeSynapse(runners.get(i).lm, alpha);
        }
        return new Pair<Double, LMSaver>(entropySum / cores, outputSaver);
    }

    private List<File> split() throws IOException {

        List<String> lines = new ArrayList<String>();
        List<File> splittedFiles = new ArrayList<File>();
        String line;
        BufferedReader reader = new BufferedReader(new FileReader(conf.trainFile));
        while ((line = reader.readLine()) != null) {
            lines.add(line);
        }
        reader.close();

        // Split files
        Random rand = new Random();
        Collections.shuffle(lines, rand);
        int numWritedLines = 0;
        for (int i = 0; i < cores; ++i) {
            int numToWrite;
            if (i == cores - 1) {
                numToWrite = lines.size() - numWritedLines;
            }
            else {
                numToWrite = lines.size() / cores;
            }
            String tmpname = String.format("%x.part", rand.nextInt());
            File tmpfile = new File(SPLIT_PATH, tmpname);
            BufferedWriter writer = new BufferedWriter(new FileWriter(tmpfile));
            for (int lineNum = numWritedLines; lineNum < numWritedLines + numToWrite; ++ lineNum) {
                writer.write(lines.get(lineNum));
                writer.write("\n");
            }
            numWritedLines += numToWrite;
            writer.close();
            splittedFiles.add(tmpfile);
        }
        return splittedFiles;
    }
}
