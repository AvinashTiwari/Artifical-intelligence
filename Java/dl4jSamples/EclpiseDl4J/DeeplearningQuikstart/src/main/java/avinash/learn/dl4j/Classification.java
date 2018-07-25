package avinash.learn.dl4j;

import org.apache.log4j.BasicConfigurator;
import org.datavec.api.io.converters.SelfWritableConverter;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxTimeIterationTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Arrays;
import java.util.concurrent.TimeUnit;

public class Classification {
    private final static Logger LOGGER = org.slf4j.LoggerFactory.getLogger(Classification.class);

	public static void main(String[] args) throws FileNotFoundException, IOException, InterruptedException {
		int batchSize = 1024;
        int labelIndex = 6;
        int numPossibleLabels = 2;
       
		BasicConfigurator.configure();
        NormalizerMinMaxScaler normalizer = new NormalizerMinMaxScaler(0, 1);
        CSVRecordReader trainReader = new CSVRecordReader(1, ",");
        trainReader.initialize(new FileSplit(new File("F:\\AI\\Java\\dl4jSamples\\EclpiseDl4J\\DeeplearningQuikstart\\data\\classification\\train.csv")));
        RecordReaderDataSetIterator trainIter = new RecordReaderDataSetIterator(trainReader, new SelfWritableConverter(), batchSize, labelIndex, numPossibleLabels, false);

        normalizer.fit(trainIter);
        trainIter.setPreProcessor(normalizer);

        CSVRecordReader testReader = new CSVRecordReader(1, ",");
        testReader.initialize(new FileSplit(new File("F:\\AI\\Java\\dl4jSamples\\EclpiseDl4J\\DeeplearningQuikstart\\data\\classification\\test.csv")));
        DataSetIterator testIter = new RecordReaderDataSetIterator(testReader, new SelfWritableConverter(), batchSize, labelIndex, numPossibleLabels, false);
        testIter.setPreProcessor(normalizer);
        
        int seed = 1990;
        double learningRate = 1e-3;
        int numInput = trainIter.inputColumns();
        int numOutputs = 2;
        int nHidden = 10;
        int epoch = 10;
        int iterations = 1;
        MultiLayerNetwork net =
                new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
                                              .seed(seed)
                                              .iterations(iterations)
                                              .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                                              .dropOut(0.5)
                                              .learningRate(learningRate)
                                              .weightInit(WeightInit.XAVIER)
                                              .updater(Updater.NESTEROVS).momentum(0.99)
                                              .list()
                                              .layer(0, new DenseLayer.Builder().nIn(numInput).nOut(nHidden)
                                                                                .activation(Activation.RELU)
                                                                                .build())
                                              .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                                      .activation(Activation.SOFTMAX)
                                                      .nIn(nHidden).nOut(numOutputs).build())
                                              .pretrain(false).backprop(true).build());
        net.init();
        net.setListeners(new ScoreIterationListener(1000));

        EarlyStoppingConfiguration<MultiLayerNetwork> esConf = new EarlyStoppingConfiguration.Builder<MultiLayerNetwork>()
                .epochTerminationConditions(new MaxEpochsTerminationCondition(5))
                .iterationTerminationConditions(new MaxTimeIterationTerminationCondition(20, TimeUnit.MINUTES))
                .scoreCalculator(new DataSetLossCalculator(testIter, true))
                .evaluateEveryNEpochs(1)
                .modelSaver(new LocalFileModelSaver("bestmodel"))
                .build();
        EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf, net, trainIter);
        EarlyStoppingResult<MultiLayerNetwork> result = trainer.fit();

        System.out.println("Termination reason: " + result.getTerminationReason());
        System.out.println("Termination details: " + result.getTerminationDetails());
        System.out.println("Total epochs: " + result.getTotalEpochs());
        System.out.println("Best epoch number: " + result.getBestModelEpoch());
        System.out.println("Score at best epoch: " + result.getBestModelScore());

        
	}

}
