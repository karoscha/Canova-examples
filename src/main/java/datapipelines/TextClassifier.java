package datapipelines;

import org.canova.api.conf.Configuration;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.impl.CollectionRecordReader;
import org.canova.api.records.reader.impl.FileRecordReader;
import org.canova.api.split.FileSplit;
import org.canova.api.writable.Writables;
import org.canova.nd4j.nlp.reader.TfidfRecordReader;
import org.canova.nd4j.nlp.vectorizer.TfidfVectorizer;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Classifier;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.LayerFactory;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.util.FeatureUtil;
import spire.macros.Auto;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;

/**
 * Created by sogolmoshtaghi on 5/6/15.
 */
public class TextClassifier {


    public static void main(String[] args) throws Exception{

        // Path to the train corpus
        String LabeledPath = System.getProperty("user.home")
                +"/dsr/data/mini_newsgroups";

        RecordReader recordReader = new TfidfRecordReader();
        recordReader.initialize(new FileSplit(new File(LabeledPath)));

         DataSetIterator iter = new RecordReaderDataSetIterator(recordReader,100,-1,3);

        // Configuring classifier
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder().layer(new OutputLayer())
                .activationFunction("softmax").weightInit(WeightInit.ZERO)
                .lossFunction(LossFunctions.LossFunction.MCXENT)
                .optimizationAlgo(OptimizationAlgorithm.GRADIENT_DESCENT)
                .nIn(iter.inputColumns()).nOut(3).build();
        Classifier classifier = (Classifier) LayerFactories.getFactory(conf).create(conf);


        // Iterating over the data and training the model
        while (iter.hasNext()) {
            DataSet dataset = iter.next();
            dataset.normalizeZeroMeanZeroUnitVariance();
            classifier.fit(dataset);
        }


//        DataSetIterator testIterator = testSet;
//        val testData = testIterator.next()
//        DataSet testData = testSet.get(0);
//        testData.normalizeZeroMeanZeroUnitVariance();
//        Evaluation eval = new Evaluation();
//        int[] output = classifier.predict(testData.getFeatureMatrix());
//        System.out.println(output);
//
//        INDArray intLabels = FeatureUtil.toOutcomeVector(0,2);
//        System.out.println("WTF?" + output);

        /**
        // training the model with each batch
        layer.fit(iter.next());
        //output is for new examples
        // INDArray output = logistic.output(X_test);
        INDArray trueLabels = Nd4j.create(5);
        INDArray intLabels = FeatureUtil.toOutcomeVector(0,2);
        Evaluation eval = new Evaluation();
        //eval.eval(trueLabels,output);
        System.out.println(eval.stats());
        //pass in to eval.eval true labels and guesses from logistic.output
      **/

    }
}
