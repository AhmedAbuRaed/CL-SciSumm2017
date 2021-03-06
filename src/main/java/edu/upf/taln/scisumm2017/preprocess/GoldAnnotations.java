package edu.upf.taln.scisumm2017.preprocess;

import edu.upf.taln.scisumm2017.Utilities;
import edu.upf.taln.scisumm2017.reader.SciSummAnnotation;
import gate.AnnotationSet;
import gate.Document;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;

/**
 * Created by Ahmed on 6/7/17.
 */
public class GoldAnnotations {
    public static HashMap<String, Document> run(HashMap<String, Document> rawDocuments, File folder, boolean isTrain) {
        ArrayList<SciSummAnnotation> sciSummAnnotations = Utilities.extractSciSummAnnotationsFromBaseFolder(folder, isTrain);
        //clear old Annotations
        for (String key : rawDocuments.keySet()) {
            Document document = rawDocuments.get(key);
            if (key.equals(folder.getName())) {
                AnnotationSet references = document.getAnnotations("REFERENCES");
                references.clear();
            } else {
                AnnotationSet citations = document.getAnnotations("CITATIONS");
                citations.clear();
            }
            rawDocuments.put(key, document);
        }
        return Utilities.applySciSummAnnotations(rawDocuments, sciSummAnnotations, isTrain);
    }
}
