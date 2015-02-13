package jrnnlm.core.scanner;

import java.io.File;
import java.io.IOException;

import jrnnlm.core.Vocabulary;
import jrnnlm.io.FileInputStreamFactory;

public class FileScanner extends StreamScanner {

    public FileScanner(Vocabulary vocab, File file) throws IOException {
      super(vocab, new FileInputStreamFactory(file));
    }
}
