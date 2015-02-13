package jrnnlm.io;

import java.io.IOException;
import java.io.InputStream;

public interface InputStreamFactory {
  public InputStream getInputStream() throws IOException;
}
