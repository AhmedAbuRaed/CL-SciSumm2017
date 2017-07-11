package edu.upf.taln.scisumm2017.process;

import edu.upf.taln.scisumm2017.Utilities;
import gate.Annotation;
import gate.AnnotationSet;
import gate.Document;

import java.io.File;
import java.util.HashMap;
import java.util.Iterator;

/**
 * Created by ahmed on 7/9/2017.
 */
public class ProcessAsTrainingPipeline {
    public static void ProcessAsTraining(String workingDirectory, String datasetType, String target)
    {
        System.out.println("Started Training Pipeline ...");
        if (target.equals("ALL")) {
            File corpus = new File(workingDirectory + "/datasets/" + datasetType);

            for (File folder : corpus.listFiles()) {
                System.out.println("Processing Cluster:" + folder.getName());
                File inputFolder = new File(folder.getPath() + File.separator + folder.getName());
                System.out.println("Extracting documents ...");
                HashMap<String, Document> documents = Utilities.extractDocumentsFromBaseFolder(inputFolder);
                System.out.println("Documents Extracted ...");

                System.out.println("Applying Cosine and Babelnet Similarities to documents ...");
                documents = Utilities.applyCosineSimilarities(documents, folder);
                documents = Utilities.applyBabelnetCosineSimilarities(documents, folder);
                System.out.println("Cosine and Babelnet Similarities to documents Applied ...");

                Document rp = documents.get(folder.getName());
                AnnotationSet rpOriginalMarkups = rp.getAnnotations("Original markups");
                AnnotationSet rpReferences = rp.getAnnotations("REFERENCES");
                AnnotationSet rpSentences = rpOriginalMarkups.get("S");

                //delete old features values
                rp.getAnnotations("NO_Match_Features").clear();
                rp.getAnnotations("Match_Features").clear();
                rp.getAnnotations("Facet_Features").clear();

                for (String key : documents.keySet()) {
                    if (!key.equals(folder.getName())) {
                        Document cp = documents.get(key);

                        AnnotationSet cpOriginalMarkups = cp.getAnnotations("Original markups");
                        AnnotationSet cpCitations = cp.getAnnotations("CITATIONS");
                        AnnotationSet cpSentences = cpOriginalMarkups.get("S");

                        Long cpStartA, cpEndA, rpStartS, rpEndS;
                        Iterator cpAnnotatorsIterator = cpCitations.iterator();

                        while (cpAnnotatorsIterator.hasNext()) {
                            Annotation cpAnnotator = (Annotation) cpAnnotatorsIterator.next();
                            cpStartA = cpAnnotator.getStartNode().getOffset();
                            cpEndA = cpAnnotator.getEndNode().getOffset();

                            AnnotationSet cpCitationSentences = cpSentences.get(cpStartA);
                            if (cpCitationSentences.size() > 0) {
                                Annotation cpSentence = cpCitationSentences.iterator().next();

                                Iterator rpSentencesIterator = rpSentences.iterator();
                                while (rpSentencesIterator.hasNext()) {
                                    Annotation rpSentence = (Annotation) rpSentencesIterator.next();
                                    rpStartS = rpSentence.getStartNode().getOffset();
                                    rpEndS = rpSentence.getEndNode().getOffset();

                                        AnnotationSet rpReferenceCitations = rpReferences.get(rpStartS, rpEndS);
                                        if (rpReferenceCitations.size() > 0) {



                                        }
                                }
                            } else {
                                System.out.println("Could not find the Citance Sentence.");
                            }
                        }
                    }
                }
            }
        }
        else {

        }
        System.out.println("Training Pipeline Done ...");
    }
}
