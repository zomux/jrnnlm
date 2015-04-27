package jrnnlm.core;

import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Serializable;
import java.util.HashMap;
import java.util.Scanner;

import jrnnlm.io.InputStreamFactory;
import jrnnlm.utils.Logger;

public class Vocabulary implements Serializable {

    /**
   * 
   */
  private static final long serialVersionUID = 1149414548501597726L;
    private final HashMap<String, Integer> vocabMap;
    private int size = 0;

    public Vocabulary() {

        vocabMap = new HashMap<String, Integer>();
        addNewWord("</s>");
        addNewWord("<unk>");
    }

    public void loadRawText(InputStreamFactory isFactory) throws IOException {

        Scanner scanner = new Scanner(new InputStreamReader(isFactory.getInputStream()));
        String word;
        while (scanner.hasNext()) {
            word = scanner.next();
            addNewWord(word);
        }
        Logger.info(String.format("[Vocabulary] size = %d", size));
        scanner.close();
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
