package jrnnlm.core.scanner;

import jrnnlm.core.Vocabulary;

import java.io.*;

public class FileScanner implements WordIndexScanner {

    private final Vocabulary vocab;
    private BufferedReader reader;
    private final File file;

    public FileScanner(Vocabulary vocab, File file) throws FileNotFoundException {

        this.vocab = vocab;
        this.file = file;
        reader = new BufferedReader(new FileReader(file));
    }

    @Override
    public int next() {

        String word = null;
        try {
            word = readWord();
        } catch (IOException ignored) {}

        if (word == null) return -1;
        else return vocab.index(word);
    }



    public String readWord() throws IOException {

        int charInt = -1;
        char c;
        StringBuilder sb = new StringBuilder();
        boolean stop = false;
        while((charInt = reader.read()) != -1) {
            c = (char) charInt;
            switch (c) {
                case '\n':
                    sb.append("</s>");
                    stop = true;
                    break;
                case '\t':
                case ' ':
                    if (sb.length() > 0) stop = true;
                    break;
                default:
                    sb.append(c);
                    break;
            }
            if (stop) break;
        }

        if (sb.length() == 0) {
            return null;
        }
        else {
            return sb.toString();
        }
    }

    @Override
    public void reset() {

        try {
            reader.close();
            reader = new BufferedReader(new FileReader(file));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
