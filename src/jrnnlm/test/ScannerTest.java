package jrnnlm.test;

import jrnnlm.core.Vocabulary;
import jrnnlm.core.scanner.FileScanner;

import java.io.*;

public class ScannerTest {

    public static void main(String[] argv) throws IOException {

        File testcase =  new File("data/ptb.train.100.txt");
        Vocabulary vocab = new Vocabulary();
        FileScanner scanner = new FileScanner(vocab, testcase);
        String s;
        while((s = scanner.readWord()) != null) {
            System.out.print(s);
            System.out.print(" ");
        }
    }
}
