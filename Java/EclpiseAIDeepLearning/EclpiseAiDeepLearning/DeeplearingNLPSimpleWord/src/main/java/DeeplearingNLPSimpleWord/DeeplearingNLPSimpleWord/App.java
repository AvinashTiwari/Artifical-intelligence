package DeeplearingNLPSimpleWord.DeeplearingNLPSimpleWord;

/**
 * Hello world!
 *
 */
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.AbstractCache;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import java.util.Collection;


public class App 
{
public static void main(String[] args) throws Exception {
        
        //as usual we need a training dataset: lots of lots of sentences and words 
        String filePath = new ClassPathResource("raw_sentences.txt").getFile().getAbsolutePath();

        //strip white space before and after for each line
        SentenceIterator sentenceIterator = new BasicLineIterator(filePath);
        //split on white spaces in the line to get words
        TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

        // manual creation of VocabCache and WeightLookupTable
        VocabCache<VocabWord> cache = new AbstractCache<>();
        WeightLookupTable<VocabWord> table = new InMemoryLookupTable.Builder<VocabWord>()
        		//we will have 100 neurons in the hidden layer: so 100 is the length of the word vectors
                .vectorLength(100)
                //ADAgrad as the optimization algorithms instead of SGD
                .useAdaGrad(false)
                .cache(cache).build();

        // word2vec algorithm with Skim-Gram model
        Word2Vec wordVecNeuralNetwork = new Word2Vec.Builder()
        		//all words below this threshold frequency will be removed prior model training
                .minWordFrequency(5)
                .iterations(1)
                .epochs(10)
                //we will have 100 neurons in the hidden layer: so 100 is the length of the word vectors
                .layerSize(100)
                .seed(42)
                //consider 5 words before + 5 words after the actual one
                .windowSize(5)
                .iterate(sentenceIterator)
                .tokenizerFactory(tokenizerFactory)
                .lookupTable(table)
                .vocabCache(cache)
                .build();

        //train the model
        wordVecNeuralNetwork.fit();

        //evaluate the model - get the 5 nearest words to "week"
        Collection<String> nearestWords = wordVecNeuralNetwork.wordsNearest("week", 5);
        System.out.println("Closest words to 'week':" + nearestWords);
    }         
}
