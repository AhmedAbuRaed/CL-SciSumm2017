package edu.upf.taln.scisumm2017.preprocess;

import edu.upf.taln.scisumm2017.Utilities;
import gate.Document;
import org.apache.commons.lang3.tuple.Pair;

import java.util.HashMap;

/**
 * Created by Ahmed on 6/13/17.
 */
public class Babelfy {
    public static HashMap<String, Document> run(HashMap<String, Document> rawDocuments) {
        Integer queryCounter = 0;
        for (String key : rawDocuments.keySet()) {
            try {
                if (queryCounter + rawDocuments.get(key).getAnnotations("Original markups").get("S").size() < 20000) {

                    System.out.println(key + ": Babelfying");

                    Pair<Document, Integer> pair = Utilities.annotateBabelnetAnnotations(rawDocuments.get(key), queryCounter);

                    queryCounter = pair.getRight();
                    System.out.println("queryCounter: " + queryCounter);

                    rawDocuments.put(key, pair.getLeft());

                    System.out.println(key + ": Babelfyed");

                } else {
                    System.out.println("Finishing because the limit is reached ...");
                    System.exit(0);
                }

            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        return rawDocuments;
    }
}
