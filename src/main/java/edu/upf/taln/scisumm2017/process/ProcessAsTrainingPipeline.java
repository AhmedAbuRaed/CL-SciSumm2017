package edu.upf.taln.scisumm2017.process;

import edu.upf.taln.ml.feat.FeatureSet;
import edu.upf.taln.scisumm2017.Utilities;
import edu.upf.taln.scisumm2017.feature.context.DocumentCtx;
import edu.upf.taln.scisumm2017.reader.TrainingExample;
import gate.Annotation;
import gate.AnnotationSet;
import gate.Document;
import gate.Factory;

import java.io.*;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Set;

/**
 * Created by ahmed on 7/9/2017.
 */
public class ProcessAsTrainingPipeline {
    public static void ProcessAsTraining(String workingDirectory, String datasetType, String target) {
        System.out.println("Started Training Pipeline ...");
        String outputInstancesType = "Training";
        if (target.equals("ALL")) {
            int matchParsedInstances = 0;
            int facetParsedInstances = 0;

            HashMap<String, Set<String>> offsetsMap = Utilities.GenerateOffsetsMap(workingDirectory);

            System.out.println("Genetaring Matches FeatureSet ...");
            FeatureSet<TrainingExample, DocumentCtx> matchesFeatureSet = Utilities.generateMatchFeatureSet();
            System.out.println("Genetaring Matches FeatureSet Done ...");
            System.out.println("Genetaring Facet FeatureSet ...");
            FeatureSet<TrainingExample, DocumentCtx> facetsFeatureSet = Utilities.generateFacetFeatureSet("ALL");
            System.out.println("Genetaring Matches FeatureSet Done ...");

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

                System.out.println("Generating Matches and Facets Instances ...");

                Document rp = documents.get(folder.getName());
                AnnotationSet rpOriginalMarkups = rp.getAnnotations("Original markups");
                AnnotationSet rpReferences = rp.getAnnotations("REFERENCES");
                AnnotationSet rpSentences = rpOriginalMarkups.get("S");

                /*
                //delete old features values
                rp.getAnnotations("NO_Match_Features").clear();
                rp.getAnnotations("Match_Features").clear();
                rp.getAnnotations("Facet_Features").clear();*/

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

                                    boolean match = false;

                                    //Match Training
                                    if (offsetsMap.get(cpAnnotator.getFeatures().get("id")).contains(rpSentence.getFeatures().get("sid"))) {
                                        try {
                                            matchParsedInstances++;
                                            System.out.println("Parsing " + outputInstancesType + " match instance " + matchParsedInstances
                                                    + ": (citance: " + cp.getName() + " reference: " + rp.getName()
                                                    + " id: ref: " + rpSentence.getFeatures().get("sid")
                                                    + " cit: " + cpSentence.getFeatures().get("sid") + "):");
                                            // Set training context
                                            DocumentCtx trCtx = new DocumentCtx(rp, cp);
                                            TrainingExample te = new TrainingExample(rpSentence, cpSentence, 0);
                                            matchesFeatureSet.addElement(te, trCtx);
                                        } catch (Exception e) {
                                            System.out.println("Error generating " + outputInstancesType
                                                    + " match instance features of example "
                                                    + matchParsedInstances
                                                    + ": (citance: " + cp.getName() + " reference: " + rp.getName()
                                                    + " id: ref: " + rpSentence.getFeatures().get("sid")
                                                    + " cit: " + cpSentence.getFeatures().get("sid") + "):");
                                            e.printStackTrace();
                                        }
                                    }

                                    AnnotationSet rpReferenceCitations = rpReferences.get(rpStartS, rpEndS);
                                    if (rpReferenceCitations.size() > 0) {
                                        Iterator rpMatchReferenceCitationsIterator = rpReferenceCitations.iterator();
                                        while (rpMatchReferenceCitationsIterator.hasNext()) {
                                            Annotation rpReferenceCitation = (Annotation) rpMatchReferenceCitationsIterator.next();
                                            if (rpReferenceCitation.getFeatures().get("id")
                                                    .equals(cpAnnotator.getFeatures().get("id"))) {
                                                match = true;
                                            }
                                        }

                                        if (match) {
                                            try {
                                                matchParsedInstances++;
                                                System.out.println("Parsing " + outputInstancesType + " match instance " + matchParsedInstances
                                                        + ": (citance: " + cp.getName() + " reference: " + rp.getName()
                                                        + " id: ref: " + rpSentence.getFeatures().get("sid")
                                                        + " cit: " + cpSentence.getFeatures().get("sid") + "):");
                                                // Set training context
                                                DocumentCtx trCtx = new DocumentCtx(rp, cp);
                                                TrainingExample te = new TrainingExample(rpSentence, cpSentence, 1);
                                                matchesFeatureSet.addElement(te, trCtx);
                                            } catch (Exception e) {
                                                System.out.println("Error generating " + outputInstancesType
                                                        + " match instance features of example "
                                                        + matchParsedInstances
                                                        + ": (citance: " + cp.getName() + " reference: " + rp.getName()
                                                        + " id: ref: " + rpSentence.getFeatures().get("sid")
                                                        + " cit: " + cpSentence.getFeatures().get("sid") + "):");
                                                e.printStackTrace();
                                            }
                                        }

                                        //Facet Training
                                        Iterator rpFacetReferenceCitationsIterator = rpReferenceCitations.iterator();
                                        while (rpFacetReferenceCitationsIterator.hasNext()) {
                                            Annotation rpReferenceCitation = (Annotation) rpFacetReferenceCitationsIterator.next();
                                            if (rpReferenceCitation.getFeatures().get("id")
                                                    .equals(cpAnnotator.getFeatures().get("id"))) {

                                                try {
                                                    facetParsedInstances++;
                                                    System.out.println("Parsing " + outputInstancesType + " facet instance " + facetParsedInstances
                                                            + ": (citance: " + cp.getName() + " reference: " + rp.getName()
                                                            + " id: ref: " + rpSentence.getFeatures().get("sid")
                                                            + " cit: " + cpSentence.getFeatures().get("sid") + "):");

                                                    // Set training context
                                                    String rpReferenceCitationFacets = rpReferenceCitation.getFeatures().get("Discourse_Facet").toString();
                                                    if (rpReferenceCitationFacets.contains(",")) {
                                                        for (String facet : rpReferenceCitationFacets.trim().split(",")) {
                                                            //We have tons of Method instances so we train others
                                                            if (!facet.replaceAll("[^a-zA-Z0-9_]", "").equals("Method_Citation")) {
                                                                facet = facet.replaceAll("[^a-zA-Z0-9_]", "");
                                                                DocumentCtx trCtx = new DocumentCtx(rp, cp);
                                                                TrainingExample te = new TrainingExample(rpSentence, cpSentence, facet);
                                                                facetsFeatureSet.addElement(te, trCtx);
                                                                break;
                                                            }
                                                        }
                                                    } else {
                                                        DocumentCtx trCtx = new DocumentCtx(rp, cp);
                                                        TrainingExample te = new TrainingExample(rpSentence, cpSentence, rpReferenceCitationFacets);
                                                        facetsFeatureSet.addElement(te, trCtx);
                                                    }

                                                } catch (Exception e) {
                                                    System.out.println("Error generating " + outputInstancesType
                                                            + " facet instance features of example "
                                                            + facetParsedInstances
                                                            + ": (citance: " + cp.getName() + " reference: " + rp.getName()
                                                            + " id: ref: " + rpSentence.getFeatures().get("sid")
                                                            + " cit: " + cpSentence.getFeatures().get("sid") + "):");
                                                    e.printStackTrace();
                                                }
                                            }
                                        }
                                    }
                                }
                            } else {
                                System.out.println("Could not find the Citance Sentence.");
                            }
                        }
                        Factory.deleteResource(cp);
                    }
                }
                Factory.deleteResource(rp);
                for (String k : documents.keySet()) {
                    Factory.deleteResource(documents.get(k));
                }
                System.gc();
            }

            Utilities.FeatureSetToARFF(matchesFeatureSet, workingDirectory, "MatchTraining", "1");
            Utilities.FeatureSetToARFF(facetsFeatureSet, workingDirectory, "FacetTraining", "1");
        } else {

        }
        System.out.println("Training Pipeline Done ...");
    }

    public static void ProcessAllSentencesAsTraining(String workingDirectory, String datasetType, String target) {
        System.out.println("Started Training Pipeline ...");
        String outputInstancesType = "Training";
        if (target.equals("ALL")) {
            int matchParsedInstances = 0;
            int facetParsedInstances = 0;

            System.out.println("Genetaring Matches FeatureSet ...");
            FeatureSet<TrainingExample, DocumentCtx> matchesFeatureSet = Utilities.generateMatchFeatureSet();
            System.out.println("Genetaring Matches FeatureSet Done ...");
            System.out.println("Genetaring Facet FeatureSet ...");
            FeatureSet<TrainingExample, DocumentCtx> facetsFeatureSet = Utilities.generateFacetFeatureSet("ALL");
            System.out.println("Genetaring Matches FeatureSet Done ...");

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

                System.out.println("Generating Matches and Facets Instances ...");

                Document rp = documents.get(folder.getName());
                AnnotationSet rpOriginalMarkups = rp.getAnnotations("Original markups");
                AnnotationSet rpReferences = rp.getAnnotations("REFERENCES");
                AnnotationSet rpSentences = rpOriginalMarkups.get("S");

                /*
                //delete old features values
                rp.getAnnotations("NO_Match_Features").clear();
                rp.getAnnotations("Match_Features").clear();
                rp.getAnnotations("Facet_Features").clear();*/

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

                                    boolean match = false;

                                    //Match Training
                                    AnnotationSet rpReferenceCitations = rpReferences.get(rpStartS, rpEndS);
                                    if (rpReferenceCitations.size() > 0) {
                                        Iterator rpMatchReferenceCitationsIterator = rpReferenceCitations.iterator();
                                        while (rpMatchReferenceCitationsIterator.hasNext()) {
                                            Annotation rpReferenceCitation = (Annotation) rpMatchReferenceCitationsIterator.next();
                                            if (rpReferenceCitation.getFeatures().get("id")
                                                    .equals(cpAnnotator.getFeatures().get("id"))) {
                                                match = true;
                                            }
                                        }

                                        if (match) {
                                            try {
                                                matchParsedInstances++;
                                                System.out.println("Parsing " + outputInstancesType + " match instance " + matchParsedInstances
                                                        + ": (citance: " + cp.getName() + " reference: " + rp.getName()
                                                        + " id: ref: " + rpSentence.getFeatures().get("sid")
                                                        + " cit: " + cpSentence.getFeatures().get("sid") + "):");
                                                // Set training context
                                                DocumentCtx trCtx = new DocumentCtx(rp, cp);
                                                TrainingExample te = new TrainingExample(rpSentence, cpSentence, 1);
                                                matchesFeatureSet.addElement(te, trCtx);
                                            } catch (Exception e) {
                                                System.out.println("Error generating " + outputInstancesType
                                                        + " match instance features of example "
                                                        + matchParsedInstances
                                                        + ": (citance: " + cp.getName() + " reference: " + rp.getName()
                                                        + " id: ref: " + rpSentence.getFeatures().get("sid")
                                                        + " cit: " + cpSentence.getFeatures().get("sid") + "):");
                                                e.printStackTrace();
                                            }
                                        } else {
                                            try {
                                                matchParsedInstances++;
                                                System.out.println("Parsing " + outputInstancesType + " match instance " + matchParsedInstances
                                                        + ": (citance: " + cp.getName() + " reference: " + rp.getName()
                                                        + " id: ref: " + rpSentence.getFeatures().get("sid")
                                                        + " cit: " + cpSentence.getFeatures().get("sid") + "):");

                                                DocumentCtx trCtx = new DocumentCtx(rp, cp);
                                                TrainingExample te = new TrainingExample(rpSentence, cpSentence, 0);
                                                matchesFeatureSet.addElement(te, trCtx);
                                            } catch (Exception e) {
                                                System.out.println("Error generating " + outputInstancesType
                                                        + " match instance features of example "
                                                        + matchParsedInstances
                                                        + ": (citance: " + cp.getName() + " reference: " + rp.getName()
                                                        + " id: ref: " + rpSentence.getFeatures().get("sid")
                                                        + " cit: " + cpSentence.getFeatures().get("sid") + "):");
                                                e.printStackTrace();
                                            }
                                        }
                                    } else {
                                        try {
                                            matchParsedInstances++;
                                            System.out.println("Parsing " + outputInstancesType + " match instance " + matchParsedInstances
                                                    + ": (citance: " + cp.getName() + " reference: " + rp.getName()
                                                    + " id: ref: " + rpSentence.getFeatures().get("sid")
                                                    + " cit: " + cpSentence.getFeatures().get("sid") + "):");

                                            DocumentCtx trCtx = new DocumentCtx(rp, cp);
                                            TrainingExample te = new TrainingExample(rpSentence, cpSentence, 0);
                                            matchesFeatureSet.addElement(te, trCtx);
                                        } catch (Exception e) {
                                            System.out.println("Error generating " + outputInstancesType
                                                    + " match instance features of example "
                                                    + matchParsedInstances
                                                    + ": (citance: " + cp.getName() + " reference: " + rp.getName()
                                                    + " id: ref: " + rpSentence.getFeatures().get("sid")
                                                    + " cit: " + cpSentence.getFeatures().get("sid") + "):");
                                            e.printStackTrace();
                                        }
                                    }
/*
                                    //Facet Training
                                    Iterator rpFacetReferenceCitationsIterator = rpReferenceCitations.iterator();
                                    while (rpFacetReferenceCitationsIterator.hasNext()) {
                                        Annotation rpReferenceCitation = (Annotation) rpFacetReferenceCitationsIterator.next();
                                        if (rpReferenceCitation.getFeatures().get("id")
                                                .equals(cpAnnotator.getFeatures().get("id"))) {

                                            try {
                                                facetParsedInstances++;
                                                System.out.println("Parsing " + outputInstancesType + " facet instance " + facetParsedInstances
                                                        + ": (citance: " + cp.getName() + " reference: " + rp.getName()
                                                        + " id: ref: " + rpSentence.getFeatures().get("sid")
                                                        + " cit: " + cpSentence.getFeatures().get("sid") + "):");

                                                // Set training context
                                                String rpReferenceCitationFacets = rpReferenceCitation.getFeatures().get("Discourse_Facet").toString();
                                                if (rpReferenceCitationFacets.contains(",")) {
                                                    for (String facet : rpReferenceCitationFacets.trim().split(",")) {
                                                        //We have tons of Method instances so we train others
                                                        if (!facet.replaceAll("[^a-zA-Z0-9_]", "").equals("Method_Citation")) {
                                                            facet = facet.replaceAll("[^a-zA-Z0-9_]", "");
                                                            DocumentCtx trCtx = new DocumentCtx(rp, cp);
                                                            TrainingExample te = new TrainingExample(rpSentence, cpSentence, facet);
                                                            facetsFeatureSet.addElement(te, trCtx);
                                                            break;
                                                        }
                                                    }
                                                } else {
                                                    DocumentCtx trCtx = new DocumentCtx(rp, cp);
                                                    TrainingExample te = new TrainingExample(rpSentence, cpSentence, rpReferenceCitationFacets);
                                                    facetsFeatureSet.addElement(te, trCtx);
                                                }

                                            } catch (Exception e) {
                                                System.out.println("Error generating " + outputInstancesType
                                                        + " facet instance features of example "
                                                        + facetParsedInstances
                                                        + ": (citance: " + cp.getName() + " reference: " + rp.getName()
                                                        + " id: ref: " + rpSentence.getFeatures().get("sid")
                                                        + " cit: " + cpSentence.getFeatures().get("sid") + "):");
                                                e.printStackTrace();
                                            }
                                        }
                                    }*/

                                }
                            } else {
                                System.out.println("Could not find the Citance Sentence.");
                            }
                        }
                        Factory.deleteResource(cp);
                    }
                }
                Factory.deleteResource(rp);
                for (String k : documents.keySet()) {
                    Factory.deleteResource(documents.get(k));
                }
                System.gc();
            }

            Utilities.FeatureSetToARFF(matchesFeatureSet, workingDirectory, "MatchTesting", "1");
            //Utilities.FeatureSetToARFF(facetsFeatureSet, workingDirectory, "FacetTesting", "1");
        } else {

        }
        System.out.println("Training Pipeline Done ...");
    }
}

