package jrnnlm.core;

import jrnnlm.utils.Logger;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.HashMap;
import java.util.Scanner;

public class Vocabulary {

    private final HashMap<String, Integer> vocabMap;
    private int size = 0;

    public Vocabulary() {

        vocabMap = new HashMap<String, Integer>();
        addNewWord("</s>");
        addNewWord("<unk>");
    }

    public void loadRawText(File trainFile) throws FileNotFoundException {

        Scanner scanner = new Scanner(new FileReader(trainFile));
        String word;
        while (scanner.hasNext()) {
            word = scanner.next();
            addNewWord(word);
        }
        Logger.info(String.format("[Vocabulary] size = %d", size));
    }

    public void addNewWord(String word) {

        if (!vocabMap.containsKey(word)) {
            vocabMap.put(word, size);
            size += 1;
        }
    }

    public int index(String word) {

        if (vocabMap.containsKey(word)) {
            return vocabMap.get(word);
        }
        else {
            return 1;
        }
    }

    public int size() {

        return size;
    }
}
