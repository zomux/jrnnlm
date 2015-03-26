package jrnnlm.test;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Scanner;

import javafx.util.Pair;
import jrnnlm.core.LMSaver;
import jrnnlm.core.RNNLM;
import jrnnlm.core.scanner.StringArrayIndexScanner;

public class RNNStringArrayTest {

  public static void main(String[] args) throws FileNotFoundException, ClassNotFoundException, IOException {
    if(args.length < 1){
      System.err.println("One required argument: <rnn model dir>");
      System.exit(-1);
    }
    
    RNNLM lm = LMSaver.load(args[0]);
    StringArrayIndexScanner words = new StringArrayIndexScanner();
    Scanner in = new Scanner(System.in);
    while(in.hasNextLine()){
      words.setWords(in.nextLine().trim().split("\\s+"));
      Pair<Double,Integer> output = lm.estimate(words);
      System.out.println(String.format("Output is the pair <%f,%d>", output.getKey(), output.getValue()));
    }
    in.close();

  }

}
