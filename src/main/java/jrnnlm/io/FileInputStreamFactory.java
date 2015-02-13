package jrnnlm.io;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.InputStream;

public class FileInputStreamFactory implements InputStreamFactory {

  private File file = null;
  private FileInputStream fis = null;
  
  public FileInputStreamFactory(File f) {
    this.file = f;
  }
  
  @Override
  public InputStream getInputStream() throws FileNotFoundException {
    fis = null;
    fis = new FileInputStream(file);
    return fis;
  }

}
