package edu.upf.taln.scisumm2017.preprocess;

import edu.upf.taln.dri.lib.demo.Util;
import gate.Document;

import java.io.File;
import java.util.HashMap;

/**
 * Created by Ahmed on 6/13/17.
 */
public class DRInventor {
    public static HashMap<String, Document> run(HashMap<String, Document> rawDocuments) {
        for (String key : rawDocuments.keySet()) {
            try {
                rawDocuments.put(key, Util.enrichSentences(rawDocuments.get(key), "Original markups", "S"));
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        return rawDocuments;
    }
}
