package jrnnlm.core.scanner;

public class StringArrayIndexScanner implements WordIndexScanner {

  private String[] words = null;
  private int wordIndex = 0;
  private int charIndex = 0;
  
  public void setWords(String[] w){
    words = w;
    wordIndex = 0;
    charIndex = 0;
  }
  
  @Override
  public int next() {
    if(charIndex >= words[wordIndex].length()){
      wordIndex++;
      charIndex = 0;
    }
    
    if(wordIndex >= words.length){
      return -1;
    }
    
    return words[wordIndex].charAt(charIndex++);
  }

  @Override
  public void reset() {
    wordIndex = 0;
    charIndex = 0;
  }

}
