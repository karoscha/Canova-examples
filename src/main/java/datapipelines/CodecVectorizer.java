package datapipelines;

import org.canova.api.conf.Configuration;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.SequenceRecordReader;
import org.canova.api.split.FileSplit;
import org.canova.codec.reader.CodecRecordReader;
import org.canova.sound.recordreader.WavFileRecordReader;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.DataSet;

import java.io.File;

/**
 * Data source: http://labrosa.ee.columbia.edu/projects/coversongs/covers80/
 */
public class CodecVectorizer {

    public static void main( String[] args )  throws Exception {

        String filepath = System.getProperty("user.home")
                +"/dsr/data/youtubemp4";

        SequenceRecordReader codecRecordReader = new CodecRecordReader();

        // Setting configuration
        Configuration conf = new Configuration();
        conf.set(CodecRecordReader.RAVEL,"true");
        conf.set(CodecRecordReader.START_FRAME,"160");
        conf.set(CodecRecordReader.TOTAL_FRAMES,"500");

        // Initializing CodecRecordReader with the data path and the configuration.
        codecRecordReader.initialize(new FileSplit(new File(filepath)));
        codecRecordReader.setConf(conf);

        DataSetIterator iter = new RecordReaderDataSetIterator(codecRecordReader);

        int counter = 0;
        while(iter.hasNext()) {
            counter++;
            DataSet next = iter.next();
            System.out.println(counter);
        }

    }
}
