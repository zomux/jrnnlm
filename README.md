JRNNLM: Recurrent neural network implementation in JAVA.
===

Recurrent neural network implementation in JAVA.

The code is refactored for simplicity and extensibility.

Current version:
- No class support
- No direct connection support

How to test JRNNLM
---
#### Small size language model
```bash
java -cp "target/*:lib/*" jrnnlm.test.training.SmallFileTest
```

#### Normal size language model
```bash
java -cp "target/*:lib/*" jrnnlm.test.training.NormalFileTest
```

#### Parallel training (Still unstable)
```bash
java -cp "target/*:lib/*" jrnnlm.test.training.SmallParallelTest
```

How to train a model for specific task
---
This code example shows how to train a simple language model.
Change the parameters in RNNLMConfiguration for different tasks.

```java
RNNLMConfiguration conf = new RNNLMConfiguration();
conf.trainFile = new File("data/your_training_file");
conf.validFile = new File("data/your_validation_file");
conf.hiddenSize = 50; // Size of hidden layer
conf.maxIters = 50; // Max iterations

RNNLM lm = new RNNLM(conf);
lm.train();
```

Size of Hidden Layer
---
- < 1M words: 50 - 200
- 1m - 10m words: 200 - 300
- more words: the vocabulary size should be limited, rare words should be mapped to "<unk>"


Raphael Shu (2014)
(still in process)
