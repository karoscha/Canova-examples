package org.deeplearning4j.scala

import scala.util.control.Breaks._
import org.canova.image.recordreader.ImageRecordReader
import org.canova.api.split.FileSplit
import java.io.File
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator
import java.util
import au.com.bytecode.opencsv.{CSVReadProc, CSV}
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.deeplearning4j.nn.api.{Classifier, OptimizationAlgorithm}
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.nn.layers.factory.LayerFactories
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.dataset.api.DataSet
import org.deeplearning4j.eval.Evaluation
import scala.util.control.Breaks
import org.nd4j.linalg.api.ndarray.INDArray


object ImageClassifierExample {

  def main(args: Array[String]) {

    // Setting the file path to data
    val recordReader = new ImageRecordReader(56, 56, true)
    val home = System.getProperty("user.home")
    recordReader.initialize(new FileSplit(new File(home, "lfw")))

    // Initializing iterator with recordreader and batch size
    val iterator = new RecordReaderDataSetIterator(recordReader, 200)

    // Setting up the configuration for logistic regression TODO: Change the classifier for images
    val conf = new NeuralNetConfiguration.Builder()
      .lossFunction(LossFunctions.LossFunction.MCXENT).optimizationAlgo(OptimizationAlgorithm.GRADIENT_DESCENT)
      .activationFunction("softmax")
      .iterations(100).weightInit(WeightInit.ZERO)
      .learningRate(1e-1).nIn(4).nOut(3).layer(new org.deeplearning4j.nn.conf.layers.OutputLayer()).build()

    // Creating the layer using the specified configuration
    val classifier = LayerFactories.getFactory(conf).create(conf).asInstanceOf[Classifier]

    // Iterating over the data and training the model
    var counter = 0
    val loop = new Breaks
    loop.breakable{
      while (iterator.hasNext) {
        counter += 1
        if(counter == 67){  // TODO: is this a bug? Last batch doesn't work
          loop.break;
        }
        val dataset = iterator.next()
        dataset.normalizeZeroMeanZeroUnitVariance()
        classifier.asInstanceOf[Classifier].fit(dataset)
      }
    }

    // Assessing the model's performance on test data
    iterator.reset()
//    while(iterator.hasNext){
//      val dataset = iterator.next()
//      dataset.normalizeZeroMeanZeroUnitVariance()
//      val output: INDArray = classifier.fit(dataset)
//      val eval = new Evaluation()
//
//    }
  }

}