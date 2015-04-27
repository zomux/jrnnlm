package jrnnlm.core.scanner;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

import jrnnlm.core.Vocabulary;
import jrnnlm.io.InputStreamFactory;

public class StreamScanner implements WordIndexScanner {

    private final Vocabulary vocab;
    private BufferedReader reader;
    private final InputStreamFactory is;

    public StreamScanner(Vocabulary vocab, InputStreamFactory trainStream) throws IOException {

        this.vocab = vocab;
        this.is = trainStream;
        reader = new BufferedReader(new InputStreamReader(trainStream.getInputStream()));
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
            reader = new BufferedReader(new InputStreamReader(is.getInputStream()));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
