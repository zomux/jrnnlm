package jrnnlm.core.scanner;

public class RawScanner implements WordIndexScanner {

    private final int[] data;
    private int pointer;

    public RawScanner(int[] data) {

        this.data = data;
        pointer = 0;
    }

    @Override
    public int next() {

        if (pointer >= data.length) {
            return -1;
        }
        else {
            return data[pointer++];
        }
    }

    @Override
    public void reset() {

        pointer = 0;
    }
}
