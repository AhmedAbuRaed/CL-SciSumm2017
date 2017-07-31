package edu.upf.taln.scisumm2017.process;

import edu.upf.taln.ml.feat.FeatUtil;
import edu.upf.taln.ml.feat.FeatureSet;
import edu.upf.taln.ml.feat.exception.FeatSetConsistencyException;
import edu.upf.taln.scisumm2017.Utilities;
import edu.upf.taln.scisumm2017.feature.context.DocumentCtx;
import edu.upf.taln.scisumm2017.reader.SciSummAnnotation;
import edu.upf.taln.scisumm2017.reader.TrainingExample;
import gate.Annotation;
import gate.AnnotationSet;
import gate.Document;
import gate.Factory;
import gate.util.InvalidOffsetException;
import org.apache.commons.lang3.StringUtils;
import weka.classifiers.misc.InputMappedClassifier;
import weka.core.Instances;

import java.io.*;
import java.util.*;

/**
 * Created by ahmed on 7/9/2017.
 */
public class ProcessAsTestingPipeline {
    public static void ProcessAsTesting(String workingDirectory, String datasetType, String target) {
        System.out.println("Started Testing Pipeline ...");
        String outputInstancesType = "Testing";
        if (target.equals("ALL")) {
            int classifiedInstances = 0;
            int version = 1;

            FeatureSet<TrainingExample, DocumentCtx> facetsFeatureSet = Utilities.generateFacetFeatureSet("ALL");

            System.out.println("Loading Facet Model ...");
            File facetModel = new File(workingDirectory + File.separator + "facetModel.model");

            File trainingFacetDatastructure = new File(workingDirectory + File.separator +
                    "facetTrainingDataSet.arff");

            System.out.println("Loading InputMappedClassifiers...");
            InputMappedClassifier facetInputMappedClassifier = Utilities.loadInputMappedClassifier(facetModel, trainingFacetDatastructure);
            System.out.println("InputMappedClassifiers Loaded ...");

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

                System.out.println(folder.getName() + " Start Classification...");

                Instances trainingFacetDataset = Utilities.readDataStructure(trainingFacetDatastructure);

                System.out.println("Loading Data from Matches System ...");
                String dataFilePath = workingDirectory + File.separator + "test_voting_MJ_WE_BN.csv";
                HashMap<String, String[]> references = new HashMap<String, String[]>();

                BufferedReader reader;
                String line;
                try {
                    reader = new BufferedReader(
                            new InputStreamReader(
                                    new FileInputStream(dataFilePath), "UTF-8"));
                    reader.readLine();
                    while ((line = reader.readLine()) != null) {
                        if (!line.equals("")) {
                            String[] values = line.split(",");

                            if (values[0].equals(folder.getName())) {
                                String[] CSIDs;
                                String[] RSIDs;
                                if (values[3].contains("+")) {
                                    CSIDs = values[3].split("\\+");
                                    for (String CSID : CSIDs) {
                                        if (values[4].contains("+")) {
                                            RSIDs = values[4].split("\\+");

                                            references.put(values[0] + "_" + values[1].replaceAll("_", "-") + "_" + values[2] + "_" + CSID, RSIDs);
                                        } else {
                                            references.put(values[0] + "_" + values[1].replaceAll("_", "-") + "_" + values[2] + "_" + CSID, new String[]{values[4]});
                                        }
                                    }
                                } else {
                                    if (values[4].contains("+")) {
                                        RSIDs = values[4].split("\\+");

                                        references.put(values[0] + "_" + values[1].replaceAll("_", "-") + "_" + values[2] + "_" + values[3], RSIDs);
                                    } else {
                                        references.put(values[0] + "_" + values[1].replaceAll("_", "-") + "_" + values[2] + "_" + values[3], new String[]{values[4]});
                                    }
                                }
                            }
                        }
                    }

                    reader.close();
                } catch (UnsupportedEncodingException e) {
                    e.printStackTrace();
                } catch (FileNotFoundException e) {
                    e.printStackTrace();
                } catch (IOException e) {
                    e.printStackTrace();
                }

                System.out.println("Generating Matches and Facets Instances ...");

                HashMap<String, SciSummAnnotation> output = new HashMap<String, SciSummAnnotation>();

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

                                    String citingID = cpAnnotator.getFeatures().get("id").toString();

                                    for (String ref : references.keySet()) {
                                        int citcounter = 0;
                                        for (int i = 0; i < citingID.length(); i++) {
                                            if (citingID.charAt(i) == '_') {
                                                citcounter++;
                                            }
                                        }
                                        if (citcounter != 4) {
                                            int citIndex = StringUtils.ordinalIndexOf(citingID, "_", 3);
                                            String citPart1 = citingID.substring(0, citIndex);
                                            String citPart2 = citingID.substring(citIndex + 1);
                                            citingID = citPart1 + "-" + citPart2;
                                        }

                                        if ((citingID.split("_")[1].equals(ref.split("_")[0]) &&
                                                citingID.split("_")[2].equals(ref.split("_")[1]) &&
                                                citingID.split("_")[0].equals(ref.split("_")[2]) &&
                                                cpSentence.getFeatures().get("sid").equals(ref.split("_")[3])) &&
                                                (Arrays.asList(references.get(ref)).contains(rpSentence.getFeatures().get("sid").toString()))) {

                                            //Testing
                                            try {
                                                classifiedInstances++;
                                                System.out.println("Classifying test instance " + classifiedInstances
                                                        + ": (citance: " + cp.getName() + " reference: " + rp.getName()
                                                        + " id: ref: " + rpSentence.getFeatures().get("sid")
                                                        + " cit: " + cpSentence.getFeatures().get("sid") + "):");

                                                // Set testing context
                                                DocumentCtx trCtx = new DocumentCtx(rp, cp);
                                                TrainingExample te;

                                                te = new TrainingExample(rpSentence, cpSentence, (String) null);
                                                facetsFeatureSet.addElement(te, trCtx);

                                                Instances facetTestInstance = FeatUtil.wekaInstanceGeneration(facetsFeatureSet, outputInstancesType + " scisumm2017_v_" + version);

                                                System.out.println("Applying String to Word Vector Filter on the facet testing Dataset...");
                                                Instances facetTestInstanceSTWVBIL = Utilities.applyStringToWordVectorFilter(facetTestInstance,
                                                        "-R 206 -P bil_ -W 1000 -prune-rate -1.0 -N 0 -stemmer weka.core.stemmers.NullStemmer -stopwords-handler weka.core.stopwords.Null -M 1 -tokenizer \"weka.core.tokenizers.WordTokenizer -delimiters \\\" \\\\r\\\\n\\\\t.,;:\\\\\\'\\\\\\\"()?!\\\"\"");
                                                Instances facetTestInstanceSTWVL = Utilities.applyStringToWordVectorFilter(facetTestInstanceSTWVBIL,
                                                        "-R 206 -P l_ -W 1000 -prune-rate -1.0 -N 0 -stemmer weka.core.stemmers.NullStemmer -stopwords-handler weka.core.stopwords.Null -M 1 -tokenizer \"weka.core.tokenizers.WordTokenizer -delimiters \\\" \\\\r\\\\n\\\\t.,;:\\\\\\'\\\\\\\"()?!\\\"\"");
                                                Instances facetTestInstanceSTWVBIP = Utilities.applyStringToWordVectorFilter(facetTestInstanceSTWVL,
                                                        "-R 206 -P bip_ -W 1000 -prune-rate -1.0 -N 0 -stemmer weka.core.stemmers.NullStemmer -stopwords-handler weka.core.stopwords.Null -M 1 -tokenizer \"weka.core.tokenizers.WordTokenizer -delimiters \\\" \\\\r\\\\n\\\\t.,;:\\\\\\'\\\\\\\"()?!\\\"\"");
                                                Instances facetTestInstanceSTWVP = Utilities.applyStringToWordVectorFilter(facetTestInstanceSTWVBIP,
                                                        "-R 206 -P p_ -W 1000 -prune-rate -1.0 -N 0 -stemmer weka.core.stemmers.NullStemmer -stopwords-handler weka.core.stopwords.Null -M 1 -tokenizer \"weka.core.tokenizers.WordTokenizer -delimiters \\\" \\\\r\\\\n\\\\t.,;:\\\\\\'\\\\\\\"()?!\\\"\"");

                                                System.out.println("Filter Applied...");

                                                System.out.println("Applying Reorder Filter on the testing Dataset...");
                                                Instances facetTestInstanceRO = Utilities.applyReorderFilter(facetTestInstanceSTWVP, "-R first-205,207-last,206");

                                                facetTestInstanceRO.setClassIndex(facetTestInstanceRO.numAttributes() - 1);
                                                System.out.println("Filter Applied...");

                                                Instances classifiedFacetTestInstance = Utilities.classifyInstances(facetTestInstanceRO, facetInputMappedClassifier);

                                                if (!classifiedFacetTestInstance.instance(0).classIsMissing()) {
                                                    String predictedfacet = classifiedFacetTestInstance.instance(0).stringValue(classifiedFacetTestInstance.instance(0).numAttributes() - 1);
                                                    System.out.println("Method Facet predicted ... ");

                                                    if (output.containsKey(cp.getAnnotations("CITATIONS").get(te.getCitanceSentence().getStartNode().getOffset()).iterator().next().getFeatures().get("Citance_Number").toString())) {
                                                        SciSummAnnotation sciSummAnnotation = output.get(cp.getAnnotations("CITATIONS").get(te.getCitanceSentence().getStartNode().getOffset()).iterator().next().getFeatures().get("Citance_Number").toString());
                                                        if (!sciSummAnnotation.getCitation_Offset().contains(te.getCitanceSentence().getFeatures().get("sid").toString())) {
                                                            sciSummAnnotation.getCitation_Offset().add(te.getCitanceSentence().getFeatures().get("sid").toString());
                                                        }
                                                        if (!sciSummAnnotation.getDiscourse_Facet().contains(predictedfacet))
                                                            sciSummAnnotation.getDiscourse_Facet().add(predictedfacet);

                                                        output.put(cp.getAnnotations("CITATIONS").get(te.getCitanceSentence().getStartNode().getOffset()).iterator().next().getFeatures().get("Citance_Number").toString(), sciSummAnnotation);
                                                    } else {
                                                        SciSummAnnotation sciSummAnnotation = new SciSummAnnotation();
                                                        sciSummAnnotation.setAnnotator(cp.getAnnotations("CITATIONS").get(te.getCitanceSentence().getStartNode().getOffset()).iterator().next().getFeatures().get("Annotator").toString());
                                                        sciSummAnnotation.setCitance_Number(cp.getAnnotations("CITATIONS").get(te.getCitanceSentence().getStartNode().getOffset()).iterator().next().getFeatures().get("Citance_Number").toString());
                                                        sciSummAnnotation.setCitation_Marker(cp.getAnnotations("CITATIONS").get(te.getCitanceSentence().getStartNode().getOffset()).iterator().next().getFeatures().get("Citation_Marker").toString());
                                                        sciSummAnnotation.setCitation_Marker_Offset(cp.getAnnotations("CITATIONS").get(te.getCitanceSentence().getStartNode().getOffset()).iterator().next().getFeatures().get("Citation_Marker_Offset").toString());
                                                        sciSummAnnotation.getReference_Offset().addAll(Arrays.asList(references.get(ref)));
                                                        sciSummAnnotation.setCiting_Article(cp.getName().substring(0, cp.getName().indexOf(".")));
                                                        sciSummAnnotation.setReference_Article(rp.getName().substring(0, rp.getName().indexOf(".")));
                                                        sciSummAnnotation.setCitation_Offset(new ArrayList<String>(Arrays.asList(new String[]{te.getCitanceSentence().getFeatures().get("sid").toString()})));
                                                        sciSummAnnotation.setDiscourse_Facet(new ArrayList<String>(Arrays.asList(new String[]{predictedfacet})));

                                                        output.put(cp.getAnnotations("CITATIONS").get(te.getCitanceSentence().getStartNode().getOffset()).iterator().next().getFeatures().get("Citance_Number").toString(), sciSummAnnotation);
                                                    }
                                                }
                                            } catch (Exception e) {
                                                System.out.println("Error generating test instance "
                                                        + " instance features of example "
                                                        + classifiedInstances
                                                        + ": (citance: " + cp.getName() + " reference: " + rp.getName()
                                                        + " id: ref: " + rpSentence.getFeatures().get("sid")
                                                        + " cit: " + cpSentence.getFeatures().get("sid") + "):");
                                                e.printStackTrace();
                                            }
                                        }
                                    }
                                }
                            } else {
                                System.out.println("Could not find the Citance Sentence.");
                            }
                        }
                    }
                }
                try {
                    output = Utilities.fillOffsetsSenttences(output, documents);
                    Utilities.printSciSummOutput(output, folder.getPath() + File.separator + "output" + File.separator + folder.getName() + ".ann.txt");
                } catch (InvalidOffsetException e) {
                    e.printStackTrace();
                }
                Factory.deleteResource(rp);
                for (String k : documents.keySet()) {
                    Factory.deleteResource(documents.get(k));
                }
                System.gc();
            }
            Utilities.FeatureSetToARFF(facetsFeatureSet, workingDirectory, "FacetTesting", "1");
        } else {

        }
        System.out.println("Testing Pipeline Done ...");
    }

    public static void ProcessAsSplitTesting(String workingDirectory, String datasetType, String target) {
        System.out.println("Started Testing Pipeline ...");
        String outputInstancesType = "Testing";
        if (target.equals("ALL")) {
            int classifiedInstances = 0;
            int version = 1;

            FeatureSet<TrainingExample, DocumentCtx> methodfacetsFeatureSet = Utilities.generateFacetFeatureSet("METHOD");
            FeatureSet<TrainingExample, DocumentCtx> othersfacetsFeatureSet = Utilities.generateFacetFeatureSet("OTHERS");

            System.out.println("Loading Facet Model ...");
            File methodfacetModel = new File(workingDirectory + File.separator + "methodfacetModel.model");
            File othersfacetModel = new File(workingDirectory + File.separator + "othersfacetModel.model");

            File methodtrainingFacetDatastructure = new File(workingDirectory + File.separator +
                    "methodfacetTrainingDataSet.arff");
            File otherstrainingFacetDatastructure = new File(workingDirectory + File.separator +
                    "othersfacetTrainingDataSet.arff");

            System.out.println("Loading InputMappedClassifiers...");
            InputMappedClassifier methodfacetInputMappedClassifier = Utilities.loadInputMappedClassifier(methodfacetModel, methodtrainingFacetDatastructure);
            InputMappedClassifier othersfacetInputMappedClassifier = Utilities.loadInputMappedClassifier(othersfacetModel, otherstrainingFacetDatastructure);
            System.out.println("InputMappedClassifiers Loaded ...");

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

                System.out.println(folder.getName() + " Start Classification...");

                Instances methodtrainingFacetDataset = Utilities.readDataStructure(methodtrainingFacetDatastructure);
                Instances otherstrainingFacetDataset = Utilities.readDataStructure(otherstrainingFacetDatastructure);

                System.out.println("Loading Data from Matches System ...");
                String dataFilePath = workingDirectory + File.separator + "test_voting_MJ_WE_BN.csv";
                HashMap<String, String[]> references = new HashMap<String, String[]>();

                BufferedReader reader;
                String line;
                try {
                    reader = new BufferedReader(
                            new InputStreamReader(
                                    new FileInputStream(dataFilePath), "UTF-8"));
                    reader.readLine();
                    while ((line = reader.readLine()) != null) {
                        if (!line.equals("")) {
                            String[] values = line.split(",");

                            if (values[0].equals(folder.getName())) {
                                String[] CSIDs;
                                String[] RSIDs;
                                if (values[3].contains("+")) {
                                    CSIDs = values[3].split("\\+");
                                    for (String CSID : CSIDs) {
                                        if (values[4].contains("+")) {
                                            RSIDs = values[4].split("\\+");

                                            references.put(values[0] + "_" + values[1].replaceAll("_", "-") + "_" + values[2] + "_" + CSID, RSIDs);
                                        } else {
                                            references.put(values[0] + "_" + values[1].replaceAll("_", "-") + "_" + values[2] + "_" + CSID, new String[]{values[4]});
                                        }
                                    }
                                } else {
                                    if (values[4].contains("+")) {
                                        RSIDs = values[4].split("\\+");

                                        references.put(values[0] + "_" + values[1].replaceAll("_", "-") + "_" + values[2] + "_" + values[3], RSIDs);
                                    } else {
                                        references.put(values[0] + "_" + values[1].replaceAll("_", "-") + "_" + values[2] + "_" + values[3], new String[]{values[4]});
                                    }
                                }
                            }
                        }
                    }

                    reader.close();
                } catch (UnsupportedEncodingException e) {
                    e.printStackTrace();
                } catch (FileNotFoundException e) {
                    e.printStackTrace();
                } catch (IOException e) {
                    e.printStackTrace();
                }

                System.out.println("Generating Matches and Facets Instances ...");

                HashMap<String, SciSummAnnotation> output = new HashMap<String, SciSummAnnotation>();

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

                                    String citingID = cpAnnotator.getFeatures().get("id").toString();

                                    for (String ref : references.keySet()) {
                                        int citcounter = 0;
                                        for (int i = 0; i < citingID.length(); i++) {
                                            if (citingID.charAt(i) == '_') {
                                                citcounter++;
                                            }
                                        }
                                        if (citcounter != 4) {
                                            int citIndex = StringUtils.ordinalIndexOf(citingID, "_", 3);
                                            String citPart1 = citingID.substring(0, citIndex);
                                            String citPart2 = citingID.substring(citIndex + 1);
                                            citingID = citPart1 + "-" + citPart2;
                                        }

                                        if ((citingID.split("_")[1].equals(ref.split("_")[0]) &&
                                                citingID.split("_")[2].equals(ref.split("_")[1]) &&
                                                citingID.split("_")[0].equals(ref.split("_")[2]) &&
                                                cpSentence.getFeatures().get("sid").equals(ref.split("_")[3])) &&
                                                (Arrays.asList(references.get(ref)).contains(rpSentence.getFeatures().get("sid").toString()))) {

                                            //Testing
                                            try {
                                                classifiedInstances++;
                                                System.out.println("Classifying test instance " + classifiedInstances
                                                        + ": (citance: " + cp.getName() + " reference: " + rp.getName()
                                                        + " id: ref: " + rpSentence.getFeatures().get("sid")
                                                        + " cit: " + cpSentence.getFeatures().get("sid") + "):");

                                                // Set testing context
                                                DocumentCtx trCtx = new DocumentCtx(rp, cp);
                                                TrainingExample te;

                                                te = new TrainingExample(rpSentence, cpSentence, (String) null);
                                                methodfacetsFeatureSet.addElement(te, trCtx);

                                                Instances methodfacetTestInstance = FeatUtil.wekaInstanceGeneration(methodfacetsFeatureSet, outputInstancesType + " scisumm2017_v_" + version);

                                                System.out.println("Applying String to Word Vector Filter on the facet testing Dataset...");
                                                Instances methodfacetTestInstanceSTWVBIL = Utilities.applyStringToWordVectorFilter(methodfacetTestInstance,
                                                        "-R 206 -P bil_ -W 1000 -prune-rate -1.0 -N 0 -stemmer weka.core.stemmers.NullStemmer -stopwords-handler weka.core.stopwords.Null -M 1 -tokenizer \"weka.core.tokenizers.WordTokenizer -delimiters \\\" \\\\r\\\\n\\\\t.,;:\\\\\\'\\\\\\\"()?!\\\"\"");
                                                Instances methodfacetTestInstanceSTWVL = Utilities.applyStringToWordVectorFilter(methodfacetTestInstanceSTWVBIL,
                                                        "-R 206 -P l_ -W 1000 -prune-rate -1.0 -N 0 -stemmer weka.core.stemmers.NullStemmer -stopwords-handler weka.core.stopwords.Null -M 1 -tokenizer \"weka.core.tokenizers.WordTokenizer -delimiters \\\" \\\\r\\\\n\\\\t.,;:\\\\\\'\\\\\\\"()?!\\\"\"");
                                                Instances methodfacetTestInstanceSTWVBIP = Utilities.applyStringToWordVectorFilter(methodfacetTestInstanceSTWVL,
                                                        "-R 206 -P bip_ -W 1000 -prune-rate -1.0 -N 0 -stemmer weka.core.stemmers.NullStemmer -stopwords-handler weka.core.stopwords.Null -M 1 -tokenizer \"weka.core.tokenizers.WordTokenizer -delimiters \\\" \\\\r\\\\n\\\\t.,;:\\\\\\'\\\\\\\"()?!\\\"\"");
                                                Instances methodfacetTestInstanceSTWVP = Utilities.applyStringToWordVectorFilter(methodfacetTestInstanceSTWVBIP,
                                                        "-R 206 -P p_ -W 1000 -prune-rate -1.0 -N 0 -stemmer weka.core.stemmers.NullStemmer -stopwords-handler weka.core.stopwords.Null -M 1 -tokenizer \"weka.core.tokenizers.WordTokenizer -delimiters \\\" \\\\r\\\\n\\\\t.,;:\\\\\\'\\\\\\\"()?!\\\"\"");

                                                System.out.println("Filter Applied...");

                                                System.out.println("Applying Reorder Filter on the testing Dataset...");
                                                Instances methodfacetTestInstanceRO = Utilities.applyReorderFilter(methodfacetTestInstanceSTWVP, "-R first-205,207-last,206");

                                                methodfacetTestInstanceRO.setClassIndex(methodfacetTestInstanceRO.numAttributes() - 1);
                                                System.out.println("Filter Applied...");

                                                Instances methodclassifiedFacetTestInstance = Utilities.classifyInstances(methodfacetTestInstanceRO, methodfacetInputMappedClassifier);

                                                if (!methodclassifiedFacetTestInstance.instance(0).classIsMissing()) {
                                                    String predictedfacet = methodclassifiedFacetTestInstance.instance(0).stringValue(methodclassifiedFacetTestInstance.instance(0).numAttributes() - 1);
                                                    if (!predictedfacet.equals("Method_Citation")) {
                                                        othersfacetsFeatureSet.addElement(te, trCtx);

                                                        Instances othersfacetTestInstance = FeatUtil.wekaInstanceGeneration(othersfacetsFeatureSet, outputInstancesType + " scisumm2017_v_" + version);

                                                        System.out.println("Applying String to Word Vector Filter on the facet testing Dataset...");
                                                        Instances othersfacetTestInstanceSTWVBIL = Utilities.applyStringToWordVectorFilter(othersfacetTestInstance,
                                                                "-R 206 -P bil_ -W 1000 -prune-rate -1.0 -N 0 -stemmer weka.core.stemmers.NullStemmer -stopwords-handler weka.core.stopwords.Null -M 1 -tokenizer \"weka.core.tokenizers.WordTokenizer -delimiters \\\" \\\\r\\\\n\\\\t.,;:\\\\\\'\\\\\\\"()?!\\\"\"");
                                                        Instances othersfacetTestInstanceSTWVL = Utilities.applyStringToWordVectorFilter(othersfacetTestInstanceSTWVBIL,
                                                                "-R 206 -P l_ -W 1000 -prune-rate -1.0 -N 0 -stemmer weka.core.stemmers.NullStemmer -stopwords-handler weka.core.stopwords.Null -M 1 -tokenizer \"weka.core.tokenizers.WordTokenizer -delimiters \\\" \\\\r\\\\n\\\\t.,;:\\\\\\'\\\\\\\"()?!\\\"\"");
                                                        Instances othersfacetTestInstanceSTWVBIP = Utilities.applyStringToWordVectorFilter(othersfacetTestInstanceSTWVL,
                                                                "-R 206 -P bip_ -W 1000 -prune-rate -1.0 -N 0 -stemmer weka.core.stemmers.NullStemmer -stopwords-handler weka.core.stopwords.Null -M 1 -tokenizer \"weka.core.tokenizers.WordTokenizer -delimiters \\\" \\\\r\\\\n\\\\t.,;:\\\\\\'\\\\\\\"()?!\\\"\"");
                                                        Instances othersfacetTestInstanceSTWVP = Utilities.applyStringToWordVectorFilter(othersfacetTestInstanceSTWVBIP,
                                                                "-R 206 -P p_ -W 1000 -prune-rate -1.0 -N 0 -stemmer weka.core.stemmers.NullStemmer -stopwords-handler weka.core.stopwords.Null -M 1 -tokenizer \"weka.core.tokenizers.WordTokenizer -delimiters \\\" \\\\r\\\\n\\\\t.,;:\\\\\\'\\\\\\\"()?!\\\"\"");

                                                        System.out.println("Filter Applied...");

                                                        System.out.println("Applying Reorder Filter on the testing Dataset...");
                                                        Instances othersfacetTestInstanceRO = Utilities.applyReorderFilter(othersfacetTestInstanceSTWVP, "-R first-205,207-last,206");

                                                        othersfacetTestInstanceRO.setClassIndex(othersfacetTestInstanceRO.numAttributes() - 1);
                                                        System.out.println("Filter Applied...");

                                                        Instances othersclassifiedFacetTestInstance = Utilities.classifyInstances(othersfacetTestInstanceRO, othersfacetInputMappedClassifier);

                                                        if (!othersclassifiedFacetTestInstance.instance(0).classIsMissing()) {
                                                            String otherspredictedfacet = othersclassifiedFacetTestInstance.instance(0).stringValue(othersclassifiedFacetTestInstance.instance(0).numAttributes() - 1);
                                                            System.out.println("Method Facet predicted ... ");

                                                            if (output.containsKey(cp.getAnnotations("CITATIONS").get(te.getCitanceSentence().getStartNode().getOffset()).iterator().next().getFeatures().get("Citance_Number").toString())) {
                                                                SciSummAnnotation sciSummAnnotation = output.get(cp.getAnnotations("CITATIONS").get(te.getCitanceSentence().getStartNode().getOffset()).iterator().next().getFeatures().get("Citance_Number").toString());
                                                                if (!sciSummAnnotation.getCitation_Offset().contains(te.getCitanceSentence().getFeatures().get("sid").toString())) {
                                                                    sciSummAnnotation.getCitation_Offset().add(te.getCitanceSentence().getFeatures().get("sid").toString());
                                                                }
                                                                if (!sciSummAnnotation.getDiscourse_Facet().contains(otherspredictedfacet))
                                                                    sciSummAnnotation.getDiscourse_Facet().add(otherspredictedfacet);

                                                                output.put(cp.getAnnotations("CITATIONS").get(te.getCitanceSentence().getStartNode().getOffset()).iterator().next().getFeatures().get("Citance_Number").toString(), sciSummAnnotation);
                                                            } else {
                                                                SciSummAnnotation sciSummAnnotation = new SciSummAnnotation();
                                                                sciSummAnnotation.setAnnotator(cp.getAnnotations("CITATIONS").get(te.getCitanceSentence().getStartNode().getOffset()).iterator().next().getFeatures().get("Annotator").toString());
                                                                sciSummAnnotation.setCitance_Number(cp.getAnnotations("CITATIONS").get(te.getCitanceSentence().getStartNode().getOffset()).iterator().next().getFeatures().get("Citance_Number").toString());
                                                                sciSummAnnotation.setCitation_Marker(cp.getAnnotations("CITATIONS").get(te.getCitanceSentence().getStartNode().getOffset()).iterator().next().getFeatures().get("Citation_Marker").toString());
                                                                sciSummAnnotation.setCitation_Marker_Offset(cp.getAnnotations("CITATIONS").get(te.getCitanceSentence().getStartNode().getOffset()).iterator().next().getFeatures().get("Citation_Marker_Offset").toString());
                                                                sciSummAnnotation.getReference_Offset().addAll(Arrays.asList(references.get(ref)));
                                                                sciSummAnnotation.setCiting_Article(cp.getName().substring(0, cp.getName().indexOf(".")));
                                                                sciSummAnnotation.setReference_Article(rp.getName().substring(0, rp.getName().indexOf(".")));
                                                                sciSummAnnotation.setCitation_Offset(new ArrayList<String>(Arrays.asList(new String[]{te.getCitanceSentence().getFeatures().get("sid").toString()})));
                                                                sciSummAnnotation.setDiscourse_Facet(new ArrayList<String>(Arrays.asList(new String[]{otherspredictedfacet})));

                                                                output.put(cp.getAnnotations("CITATIONS").get(te.getCitanceSentence().getStartNode().getOffset()).iterator().next().getFeatures().get("Citance_Number").toString(), sciSummAnnotation);
                                                            }
                                                        }
                                                    } else {
                                                        System.out.println("Method Facet predicted ... ");

                                                        if (output.containsKey(cp.getAnnotations("CITATIONS").get(te.getCitanceSentence().getStartNode().getOffset()).iterator().next().getFeatures().get("Citance_Number").toString())) {
                                                            SciSummAnnotation sciSummAnnotation = output.get(cp.getAnnotations("CITATIONS").get(te.getCitanceSentence().getStartNode().getOffset()).iterator().next().getFeatures().get("Citance_Number").toString());
                                                            if (!sciSummAnnotation.getCitation_Offset().contains(te.getCitanceSentence().getFeatures().get("sid").toString())) {
                                                                sciSummAnnotation.getCitation_Offset().add(te.getCitanceSentence().getFeatures().get("sid").toString());
                                                            }
                                                            if (!sciSummAnnotation.getDiscourse_Facet().contains(predictedfacet))
                                                                sciSummAnnotation.getDiscourse_Facet().add(predictedfacet);

                                                            output.put(cp.getAnnotations("CITATIONS").get(te.getCitanceSentence().getStartNode().getOffset()).iterator().next().getFeatures().get("Citance_Number").toString(), sciSummAnnotation);
                                                        } else {
                                                            SciSummAnnotation sciSummAnnotation = new SciSummAnnotation();
                                                            sciSummAnnotation.setAnnotator(cp.getAnnotations("CITATIONS").get(te.getCitanceSentence().getStartNode().getOffset()).iterator().next().getFeatures().get("Annotator").toString());
                                                            sciSummAnnotation.setCitance_Number(cp.getAnnotations("CITATIONS").get(te.getCitanceSentence().getStartNode().getOffset()).iterator().next().getFeatures().get("Citance_Number").toString());
                                                            sciSummAnnotation.setCitation_Marker(cp.getAnnotations("CITATIONS").get(te.getCitanceSentence().getStartNode().getOffset()).iterator().next().getFeatures().get("Citation_Marker").toString());
                                                            sciSummAnnotation.setCitation_Marker_Offset(cp.getAnnotations("CITATIONS").get(te.getCitanceSentence().getStartNode().getOffset()).iterator().next().getFeatures().get("Citation_Marker_Offset").toString());
                                                            sciSummAnnotation.getReference_Offset().addAll(Arrays.asList(references.get(ref)));
                                                            sciSummAnnotation.setCiting_Article(cp.getName().substring(0, cp.getName().indexOf(".")));
                                                            sciSummAnnotation.setReference_Article(rp.getName().substring(0, rp.getName().indexOf(".")));
                                                            sciSummAnnotation.setCitation_Offset(new ArrayList<String>(Arrays.asList(new String[]{te.getCitanceSentence().getFeatures().get("sid").toString()})));
                                                            sciSummAnnotation.setDiscourse_Facet(new ArrayList<String>(Arrays.asList(new String[]{predictedfacet})));

                                                            output.put(cp.getAnnotations("CITATIONS").get(te.getCitanceSentence().getStartNode().getOffset()).iterator().next().getFeatures().get("Citance_Number").toString(), sciSummAnnotation);
                                                        }
                                                    }
                                                }
                                            } catch (Exception e) {
                                                System.out.println("Error generating test instance "
                                                        + " instance features of example "
                                                        + classifiedInstances
                                                        + ": (citance: " + cp.getName() + " reference: " + rp.getName()
                                                        + " id: ref: " + rpSentence.getFeatures().get("sid")
                                                        + " cit: " + cpSentence.getFeatures().get("sid") + "):");
                                                e.printStackTrace();
                                            }
                                        }

                                    }
                                }
                            } else {
                                System.out.println("Could not find the Citance Sentence.");
                            }
                        }
                    }
                }
                try {
                    output = Utilities.fillOffsetsSenttences(output, documents);
                    Utilities.printSciSummOutput(output, folder.getPath() + File.separator + "output" + File.separator + folder.getName() + ".ann.txt");
                } catch (InvalidOffsetException e) {
                    e.printStackTrace();
                }
                Factory.deleteResource(rp);
                for (String k : documents.keySet()) {
                    Factory.deleteResource(documents.get(k));
                }
                System.gc();
            }

            Utilities.FeatureSetToARFF(methodfacetsFeatureSet, workingDirectory, "methodFacetTesting", "1");
            Utilities.FeatureSetToARFF(othersfacetsFeatureSet, workingDirectory, "othersFacetTesting", "1");
        } else {

        }
        System.out.println("Testing Pipeline Done ...");
    }

    public static void ProcessAsSplitOneTesting(String workingDirectory, String datasetType, String target) {
        System.out.println("Started Testing Pipeline ...");
        String outputInstancesType = "Testing";
        if (target.equals("ALL")) {
            int classifiedInstances = 0;
            int version = 1;

            FeatureSet<TrainingExample, DocumentCtx> methodfacetsFeatureSet = Utilities.generateFacetFeatureSet("METHOD");
            FeatureSet<TrainingExample, DocumentCtx> othersfacetsFeatureSet = Utilities.generateFacetFeatureSet("OTHERS");

            System.out.println("Loading Facet Model ...");
            File methodfacetModel = new File(workingDirectory + File.separator + "methodfacetModel.model");
            File othersfacetModel = new File(workingDirectory + File.separator + "othersfacetModel.model");

            File methodtrainingFacetDatastructure = new File(workingDirectory + File.separator +
                    "methodfacetTrainingDataSet.arff");
            File otherstrainingFacetDatastructure = new File(workingDirectory + File.separator +
                    "othersfacetTrainingDataSet.arff");

            System.out.println("Loading InputMappedClassifiers...");
            InputMappedClassifier methodfacetInputMappedClassifier = Utilities.loadInputMappedClassifier(methodfacetModel, methodtrainingFacetDatastructure);
            InputMappedClassifier othersfacetInputMappedClassifier = Utilities.loadInputMappedClassifier(othersfacetModel, otherstrainingFacetDatastructure);
            System.out.println("InputMappedClassifiers Loaded ...");

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

                System.out.println(folder.getName() + " Start Classification...");

                Instances methodtrainingFacetDataset = Utilities.readDataStructure(methodtrainingFacetDatastructure);
                Instances otherstrainingFacetDataset = Utilities.readDataStructure(otherstrainingFacetDatastructure);

                System.out.println("Loading Data from Matches System ...");
                String dataFilePath = workingDirectory + File.separator + "test_BN.csv";
                HashMap<String, String[]> references = new HashMap<String, String[]>();

                BufferedReader reader;
                String line;
                try {
                    reader = new BufferedReader(
                            new InputStreamReader(
                                    new FileInputStream(dataFilePath), "UTF-8"));
                    reader.readLine();
                    while ((line = reader.readLine()) != null) {
                        if (!line.equals("")) {
                            String[] values = line.split(",");

                            if (values[0].equals(folder.getName())) {
                                String[] CSIDs;
                                String[] RSIDs;
                                if (values[3].contains("+")) {
                                    CSIDs = values[3].split("\\+");
                                    for (String CSID : CSIDs) {
                                        references.put(values[0] + "_" + values[1].replaceAll("_", "-") + "_" + values[2] + "_" + CSID, new String[]{values[4], values[5]});
                                    }
                                } else {
                                    references.put(values[0] + "_" + values[1].replaceAll("_", "-") + "_" + values[2] + "_" + values[3], new String[]{values[4], values[5]});
                                }
                            }
                        }
                    }

                    reader.close();
                } catch (UnsupportedEncodingException e) {
                    e.printStackTrace();
                } catch (FileNotFoundException e) {
                    e.printStackTrace();
                } catch (IOException e) {
                    e.printStackTrace();
                }

                System.out.println("Generating Matches and Facets Instances ...");

                HashMap<String, SciSummAnnotation> output = new HashMap<String, SciSummAnnotation>();

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

                                    String citingID = cpAnnotator.getFeatures().get("id").toString();

                                    for (String ref : references.keySet()) {
                                        int citcounter = 0;
                                        for (int i = 0; i < citingID.length(); i++) {
                                            if (citingID.charAt(i) == '_') {
                                                citcounter++;
                                            }
                                        }
                                        if (citcounter != 4) {
                                            int citIndex = StringUtils.ordinalIndexOf(citingID, "_", 3);
                                            String citPart1 = citingID.substring(0, citIndex);
                                            String citPart2 = citingID.substring(citIndex + 1);
                                            citingID = citPart1 + "-" + citPart2;
                                        }

                                        if ((citingID.split("_")[1].equals(ref.split("_")[0]) &&
                                                citingID.split("_")[2].equals(ref.split("_")[1]) &&
                                                citingID.split("_")[0].equals(ref.split("_")[2]) &&
                                                cpSentence.getFeatures().get("sid").equals(ref.split("_")[3])) &&
                                                (Arrays.asList(references.get(ref)).contains(rpSentence.getFeatures().get("sid").toString()))) {

                                            //Testing
                                            try {
                                                classifiedInstances++;
                                                System.out.println("Classifying test instance " + classifiedInstances
                                                        + ": (citance: " + cp.getName() + " reference: " + rp.getName()
                                                        + " id: ref: " + rpSentence.getFeatures().get("sid")
                                                        + " cit: " + cpSentence.getFeatures().get("sid") + "):");

                                                // Set testing context
                                                DocumentCtx trCtx = new DocumentCtx(rp, cp);
                                                TrainingExample te;

                                                te = new TrainingExample(rpSentence, cpSentence, (String) null);
                                                methodfacetsFeatureSet.addElement(te, trCtx);

                                                Instances methodfacetTestInstance = FeatUtil.wekaInstanceGeneration(methodfacetsFeatureSet, outputInstancesType + " scisumm2017_v_" + version);

                                                System.out.println("Applying String to Word Vector Filter on the facet testing Dataset...");
                                                Instances methodfacetTestInstanceSTWVBIL = Utilities.applyStringToWordVectorFilter(methodfacetTestInstance,
                                                        "-R 206 -P bil_ -W 1000 -prune-rate -1.0 -N 0 -stemmer weka.core.stemmers.NullStemmer -stopwords-handler weka.core.stopwords.Null -M 1 -tokenizer \"weka.core.tokenizers.WordTokenizer -delimiters \\\" \\\\r\\\\n\\\\t.,;:\\\\\\'\\\\\\\"()?!\\\"\"");
                                                Instances methodfacetTestInstanceSTWVL = Utilities.applyStringToWordVectorFilter(methodfacetTestInstanceSTWVBIL,
                                                        "-R 206 -P l_ -W 1000 -prune-rate -1.0 -N 0 -stemmer weka.core.stemmers.NullStemmer -stopwords-handler weka.core.stopwords.Null -M 1 -tokenizer \"weka.core.tokenizers.WordTokenizer -delimiters \\\" \\\\r\\\\n\\\\t.,;:\\\\\\'\\\\\\\"()?!\\\"\"");
                                                Instances methodfacetTestInstanceSTWVBIP = Utilities.applyStringToWordVectorFilter(methodfacetTestInstanceSTWVL,
                                                        "-R 206 -P bip_ -W 1000 -prune-rate -1.0 -N 0 -stemmer weka.core.stemmers.NullStemmer -stopwords-handler weka.core.stopwords.Null -M 1 -tokenizer \"weka.core.tokenizers.WordTokenizer -delimiters \\\" \\\\r\\\\n\\\\t.,;:\\\\\\'\\\\\\\"()?!\\\"\"");
                                                Instances methodfacetTestInstanceSTWVP = Utilities.applyStringToWordVectorFilter(methodfacetTestInstanceSTWVBIP,
                                                        "-R 206 -P p_ -W 1000 -prune-rate -1.0 -N 0 -stemmer weka.core.stemmers.NullStemmer -stopwords-handler weka.core.stopwords.Null -M 1 -tokenizer \"weka.core.tokenizers.WordTokenizer -delimiters \\\" \\\\r\\\\n\\\\t.,;:\\\\\\'\\\\\\\"()?!\\\"\"");

                                                System.out.println("Filter Applied...");

                                                System.out.println("Applying Reorder Filter on the testing Dataset...");
                                                Instances methodfacetTestInstanceRO = Utilities.applyReorderFilter(methodfacetTestInstanceSTWVP, "-R first-205,207-last,206");

                                                methodfacetTestInstanceRO.setClassIndex(methodfacetTestInstanceRO.numAttributes() - 1);
                                                System.out.println("Filter Applied...");

                                                Instances methodclassifiedFacetTestInstance = Utilities.classifyInstances(methodfacetTestInstanceRO, methodfacetInputMappedClassifier);

                                                if (!methodclassifiedFacetTestInstance.instance(0).classIsMissing()) {
                                                    String predictedfacet = methodclassifiedFacetTestInstance.instance(0).stringValue(methodclassifiedFacetTestInstance.instance(0).numAttributes() - 1);
                                                    if (!predictedfacet.equals("Method_Citation")) {
                                                        othersfacetsFeatureSet.addElement(te, trCtx);

                                                        Instances othersfacetTestInstance = FeatUtil.wekaInstanceGeneration(othersfacetsFeatureSet, outputInstancesType + " scisumm2017_v_" + version);

                                                        System.out.println("Applying String to Word Vector Filter on the facet testing Dataset...");
                                                        Instances othersfacetTestInstanceSTWVBIL = Utilities.applyStringToWordVectorFilter(othersfacetTestInstance,
                                                                "-R 206 -P bil_ -W 1000 -prune-rate -1.0 -N 0 -stemmer weka.core.stemmers.NullStemmer -stopwords-handler weka.core.stopwords.Null -M 1 -tokenizer \"weka.core.tokenizers.WordTokenizer -delimiters \\\" \\\\r\\\\n\\\\t.,;:\\\\\\'\\\\\\\"()?!\\\"\"");
                                                        Instances othersfacetTestInstanceSTWVL = Utilities.applyStringToWordVectorFilter(othersfacetTestInstanceSTWVBIL,
                                                                "-R 206 -P l_ -W 1000 -prune-rate -1.0 -N 0 -stemmer weka.core.stemmers.NullStemmer -stopwords-handler weka.core.stopwords.Null -M 1 -tokenizer \"weka.core.tokenizers.WordTokenizer -delimiters \\\" \\\\r\\\\n\\\\t.,;:\\\\\\'\\\\\\\"()?!\\\"\"");
                                                        Instances othersfacetTestInstanceSTWVBIP = Utilities.applyStringToWordVectorFilter(othersfacetTestInstanceSTWVL,
                                                                "-R 206 -P bip_ -W 1000 -prune-rate -1.0 -N 0 -stemmer weka.core.stemmers.NullStemmer -stopwords-handler weka.core.stopwords.Null -M 1 -tokenizer \"weka.core.tokenizers.WordTokenizer -delimiters \\\" \\\\r\\\\n\\\\t.,;:\\\\\\'\\\\\\\"()?!\\\"\"");
                                                        Instances othersfacetTestInstanceSTWVP = Utilities.applyStringToWordVectorFilter(othersfacetTestInstanceSTWVBIP,
                                                                "-R 206 -P p_ -W 1000 -prune-rate -1.0 -N 0 -stemmer weka.core.stemmers.NullStemmer -stopwords-handler weka.core.stopwords.Null -M 1 -tokenizer \"weka.core.tokenizers.WordTokenizer -delimiters \\\" \\\\r\\\\n\\\\t.,;:\\\\\\'\\\\\\\"()?!\\\"\"");

                                                        System.out.println("Filter Applied...");

                                                        System.out.println("Applying Reorder Filter on the testing Dataset...");
                                                        Instances othersfacetTestInstanceRO = Utilities.applyReorderFilter(othersfacetTestInstanceSTWVP, "-R first-205,207-last,206");

                                                        othersfacetTestInstanceRO.setClassIndex(othersfacetTestInstanceRO.numAttributes() - 1);
                                                        System.out.println("Filter Applied...");

                                                        Instances othersclassifiedFacetTestInstance = Utilities.classifyInstances(othersfacetTestInstanceRO, othersfacetInputMappedClassifier);

                                                        if (!othersclassifiedFacetTestInstance.instance(0).classIsMissing()) {
                                                            String otherspredictedfacet = othersclassifiedFacetTestInstance.instance(0).stringValue(othersclassifiedFacetTestInstance.instance(0).numAttributes() - 1);
                                                            System.out.println("Method Facet predicted ... ");

                                                            if (output.containsKey(cp.getAnnotations("CITATIONS").get(te.getCitanceSentence().getStartNode().getOffset()).iterator().next().getFeatures().get("Citance_Number").toString())) {
                                                                SciSummAnnotation sciSummAnnotation = output.get(cp.getAnnotations("CITATIONS").get(te.getCitanceSentence().getStartNode().getOffset()).iterator().next().getFeatures().get("Citance_Number").toString());
                                                                if (!sciSummAnnotation.getCitation_Offset().contains(te.getCitanceSentence().getFeatures().get("sid").toString())) {
                                                                    sciSummAnnotation.getCitation_Offset().add(te.getCitanceSentence().getFeatures().get("sid").toString());
                                                                }
                                                                if (!sciSummAnnotation.getDiscourse_Facet().contains(otherspredictedfacet))
                                                                    sciSummAnnotation.getDiscourse_Facet().add(otherspredictedfacet);

                                                                output.put(cp.getAnnotations("CITATIONS").get(te.getCitanceSentence().getStartNode().getOffset()).iterator().next().getFeatures().get("Citance_Number").toString(), sciSummAnnotation);
                                                            } else {
                                                                SciSummAnnotation sciSummAnnotation = new SciSummAnnotation();
                                                                sciSummAnnotation.setAnnotator(cp.getAnnotations("CITATIONS").get(te.getCitanceSentence().getStartNode().getOffset()).iterator().next().getFeatures().get("Annotator").toString());
                                                                sciSummAnnotation.setCitance_Number(cp.getAnnotations("CITATIONS").get(te.getCitanceSentence().getStartNode().getOffset()).iterator().next().getFeatures().get("Citance_Number").toString());
                                                                sciSummAnnotation.setCitation_Marker(cp.getAnnotations("CITATIONS").get(te.getCitanceSentence().getStartNode().getOffset()).iterator().next().getFeatures().get("Citation_Marker").toString());
                                                                sciSummAnnotation.setCitation_Marker_Offset(cp.getAnnotations("CITATIONS").get(te.getCitanceSentence().getStartNode().getOffset()).iterator().next().getFeatures().get("Citation_Marker_Offset").toString());
                                                                sciSummAnnotation.getReference_Offset().addAll(Arrays.asList(references.get(ref)));
                                                                sciSummAnnotation.setCiting_Article(cp.getName().substring(0, cp.getName().indexOf(".")));
                                                                sciSummAnnotation.setReference_Article(rp.getName().substring(0, rp.getName().indexOf(".")));
                                                                sciSummAnnotation.setCitation_Offset(new ArrayList<String>(Arrays.asList(new String[]{te.getCitanceSentence().getFeatures().get("sid").toString()})));
                                                                sciSummAnnotation.setDiscourse_Facet(new ArrayList<String>(Arrays.asList(new String[]{otherspredictedfacet})));

                                                                output.put(cp.getAnnotations("CITATIONS").get(te.getCitanceSentence().getStartNode().getOffset()).iterator().next().getFeatures().get("Citance_Number").toString(), sciSummAnnotation);
                                                            }
                                                        }
                                                    } else {
                                                        System.out.println("Method Facet predicted ... ");

                                                        if (output.containsKey(cp.getAnnotations("CITATIONS").get(te.getCitanceSentence().getStartNode().getOffset()).iterator().next().getFeatures().get("Citance_Number").toString())) {
                                                            SciSummAnnotation sciSummAnnotation = output.get(cp.getAnnotations("CITATIONS").get(te.getCitanceSentence().getStartNode().getOffset()).iterator().next().getFeatures().get("Citance_Number").toString());
                                                            if (!sciSummAnnotation.getCitation_Offset().contains(te.getCitanceSentence().getFeatures().get("sid").toString())) {
                                                                sciSummAnnotation.getCitation_Offset().add(te.getCitanceSentence().getFeatures().get("sid").toString());
                                                            }
                                                            if (!sciSummAnnotation.getDiscourse_Facet().contains(predictedfacet))
                                                                sciSummAnnotation.getDiscourse_Facet().add(predictedfacet);

                                                            output.put(cp.getAnnotations("CITATIONS").get(te.getCitanceSentence().getStartNode().getOffset()).iterator().next().getFeatures().get("Citance_Number").toString(), sciSummAnnotation);
                                                        } else {
                                                            SciSummAnnotation sciSummAnnotation = new SciSummAnnotation();
                                                            sciSummAnnotation.setAnnotator(cp.getAnnotations("CITATIONS").get(te.getCitanceSentence().getStartNode().getOffset()).iterator().next().getFeatures().get("Annotator").toString());
                                                            sciSummAnnotation.setCitance_Number(cp.getAnnotations("CITATIONS").get(te.getCitanceSentence().getStartNode().getOffset()).iterator().next().getFeatures().get("Citance_Number").toString());
                                                            sciSummAnnotation.setCitation_Marker(cp.getAnnotations("CITATIONS").get(te.getCitanceSentence().getStartNode().getOffset()).iterator().next().getFeatures().get("Citation_Marker").toString());
                                                            sciSummAnnotation.setCitation_Marker_Offset(cp.getAnnotations("CITATIONS").get(te.getCitanceSentence().getStartNode().getOffset()).iterator().next().getFeatures().get("Citation_Marker_Offset").toString());
                                                            sciSummAnnotation.getReference_Offset().addAll(Arrays.asList(references.get(ref)));
                                                            sciSummAnnotation.setCiting_Article(cp.getName().substring(0, cp.getName().indexOf(".")));
                                                            sciSummAnnotation.setReference_Article(rp.getName().substring(0, rp.getName().indexOf(".")));
                                                            sciSummAnnotation.setCitation_Offset(new ArrayList<String>(Arrays.asList(new String[]{te.getCitanceSentence().getFeatures().get("sid").toString()})));
                                                            sciSummAnnotation.setDiscourse_Facet(new ArrayList<String>(Arrays.asList(new String[]{predictedfacet})));

                                                            output.put(cp.getAnnotations("CITATIONS").get(te.getCitanceSentence().getStartNode().getOffset()).iterator().next().getFeatures().get("Citance_Number").toString(), sciSummAnnotation);
                                                        }
                                                    }
                                                }
                                            } catch (Exception e) {
                                                System.out.println("Error generating test instance "
                                                        + " instance features of example "
                                                        + classifiedInstances
                                                        + ": (citance: " + cp.getName() + " reference: " + rp.getName()
                                                        + " id: ref: " + rpSentence.getFeatures().get("sid")
                                                        + " cit: " + cpSentence.getFeatures().get("sid") + "):");
                                                e.printStackTrace();
                                            }
                                        }

                                    }
                                }
                            } else {
                                System.out.println("Could not find the Citance Sentence.");
                            }
                        }
                    }
                }
                try {
                    output = Utilities.fillOffsetsSenttences(output, documents);
                    Utilities.printSciSummOutput(output, folder.getPath() + File.separator + "output" + File.separator + folder.getName() + ".ann.txt");
                } catch (InvalidOffsetException e) {
                    e.printStackTrace();
                }
                Factory.deleteResource(rp);
                for (String k : documents.keySet()) {
                    Factory.deleteResource(documents.get(k));
                }
                System.gc();
            }

            Utilities.FeatureSetToARFF(methodfacetsFeatureSet, workingDirectory, "methodFacetTesting", "1");
            Utilities.FeatureSetToARFF(othersfacetsFeatureSet, workingDirectory, "othersFacetTesting", "1");
        } else {

        }
        System.out.println("Testing Pipeline Done ...");
    }

}
