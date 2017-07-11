package edu.upf.taln.scisumm2017.preprocess;

import edu.upf.taln.dri.lib.demo.Util;
import edu.upf.taln.scisumm2017.Utilities;
import gate.*;
import org.apache.commons.lang3.tuple.Pair;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.HashMap;
import java.util.Iterator;


/**
 * Created by Ahmed on 6/12/17.
 */
public class WordEmbedding {

    public static HashMap<String, Document> run(HashMap<String, Document> rawDocuments, Word2Vec word2Vec, String annotationSetName) {

        for (String key : rawDocuments.keySet()) {
            try {
                System.out.println("Running " + annotationSetName + " Word Embedding on Document: " + key);
                rawDocuments.put(key, Utilities.annotateWord2VecAnnotations(rawDocuments.get(key), word2Vec, annotationSetName));
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        return rawDocuments;
    }
}
