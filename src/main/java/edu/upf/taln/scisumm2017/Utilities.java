package edu.upf.taln.scisumm2017;

import edu.upf.taln.dri.lib.model.ext.Sentence;
import edu.upf.taln.ml.feat.*;
import edu.upf.taln.ml.feat.exception.FeatSetConsistencyException;
import edu.upf.taln.ml.feat.exception.FeatureException;
import edu.upf.taln.scisumm2017.feature.calculator.*;
import edu.upf.taln.scisumm2017.feature.context.DocumentCtx;
import edu.upf.taln.scisumm2017.reader.BabelnetSynset;
import edu.upf.taln.scisumm2017.reader.SciSummAnnotation;
import edu.upf.taln.scisumm2017.reader.TrainingExample;
import gate.AnnotationSet;
import gate.Factory;
import gate.Document;
import gate.Annotation;
import gate.FeatureMap;
import gate.creole.ResourceInstantiationException;
import gate.util.InvalidOffsetException;
import gate.util.OffsetComparator;

import org.apache.commons.lang3.tuple.Pair;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.deeplearning4j.models.word2vec.Word2Vec;
import weka.classifiers.misc.InputMappedClassifier;
import weka.classifiers.misc.SerializedClassifier;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Reorder;
import weka.filters.unsupervised.attribute.StringToWordVector;

import java.io.*;
import java.net.MalformedURLException;
import java.net.URL;
import java.net.URLEncoder;
import java.util.*;

/**
 * Created by Ahmed on 6/7/17.
 */
public class Utilities {
    public static HashMap<String, Document> extractDocumentsFromBaseFolder(File referenceBaseFolder) {
        Document doc = null;
        HashMap<String, Document> documents = new HashMap<String, Document>();

        try {
            for (File document : referenceBaseFolder.listFiles()) {
                doc = Factory.newDocument(new URL("file:///" + document.getPath()), "UTF-8");
                documents.put(document.getName().substring(0, document.getName().indexOf(".")), doc);
            }
        } catch (MalformedURLException e) {
            e.printStackTrace();
        } catch (ResourceInstantiationException e) {
            e.printStackTrace();
        }
        return documents;
    }

    public static ArrayList<SciSummAnnotation> extractSciSummAnnotationsFromBaseFolder(File referenceBaseFolder,
                                                                                       boolean generateTraining) {
        ArrayList<SciSummAnnotation> annotationsList = new ArrayList<SciSummAnnotation>();

        String annotationsFilePath = referenceBaseFolder + "/annotation/" +
                referenceBaseFolder.getName() + ".ann.txt";
        BufferedReader reader;
        String line;
        try {
            reader = new BufferedReader(
                    new InputStreamReader(
                            new FileInputStream(annotationsFilePath), "UTF-8"));

            while ((line = reader.readLine()) != null) {
                if (!line.equals("")) {
                    String[] fields = line.split("\\|");
                    System.out.println(fields[0]);
                    SciSummAnnotation annotation = new SciSummAnnotation();
                    annotation.setCitance_Number(fields[0].trim().split(":")[1].trim());
                    if (fields[1].trim().split(":")[1].trim().contains(".")) {
                        annotation.setReference_Article(fields[1].trim().split(":")[1].trim().substring(0, fields[1].trim().split(":")[1].trim().lastIndexOf('.')));
                    } else {
                        annotation.setReference_Article(fields[1].trim().split(":")[1].trim());
                    }
                    if (fields[2].trim().split(":")[1].trim().contains(".")) {
                        annotation.setCiting_Article(fields[2].trim().split(":")[1].trim().substring(0, fields[2].trim().split(":")[1].trim().lastIndexOf('.')));
                    } else {
                        annotation.setCiting_Article(fields[2].trim().split(":")[1].trim());
                    }
                    annotation.setCitation_Marker_Offset(fields[3].trim().split(":")[1].trim().replaceAll("\\D+", ""));
                    annotation.setCitation_Marker(fields[4].trim().split(":")[1].trim());
                    for (String co : fields[5].trim().split(":")[1].trim().split(",")) {
                        annotation.getCitation_Offset().add(co.replaceAll("\\D+", ""));
                    }
                    annotation.setCitation_Text(fields[6].trim().split(":")[1].trim());

                    if (generateTraining) {
                        for (String ro : fields[7].trim().split(":")[1].trim().split(",")) {
                            annotation.getReference_Offset().add(ro.replaceAll("\\D+", ""));
                        }
                        annotation.setReference_Text(fields[8].trim().split(":")[1].trim());
                        for (String facet : fields[9].trim().split(":")[1].trim().split(",")) {
                            annotation.getDiscourse_Facet().add(facet.replaceAll("[^a-zA-Z0-9_]", "").trim());
                        }
                        if (fields[10].trim().split(":")[1].trim().contains(",")) {
                            annotation.setAnnotator(fields[10].trim().split(":")[1].trim()
                                    .substring(0, fields[10].trim().split(":")[1].trim().indexOf(",")).replaceAll(" ", "_"));
                        } else {
                            annotation.setAnnotator(fields[10].trim().split(":")[1].trim().replaceAll(" ", "_"));
                        }
                    } else {
                        if (fields[7].trim().split(":")[1].trim().contains(",")) {
                            annotation.setAnnotator(fields[7].trim().split(":")[1].trim()
                                    .substring(0, fields[7].trim().split(":")[1].trim().indexOf(",")).replaceAll(" ", "_"));
                        } else {
                            annotation.setAnnotator(fields[7].trim().split(":")[1].trim().replaceAll(" ", "_"));
                        }
                    }
                    annotationsList.add(annotation);
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
        return annotationsList;
    }

    public static HashMap<String, Document> applySciSummAnnotations
            (HashMap<String, Document> RCDocuments, ArrayList<SciSummAnnotation> SciSummAnnotationsList,
             boolean generateTraining) {
        HashMap<String, Document> processedDocuments = new HashMap<String, Document>();
        ArrayList<String> reference_Offset = new ArrayList<>();
        String reference_Text = null;
        String annotator = null;
        ArrayList<String> discourse_Facet = new ArrayList<String>();
        Document rp = null;
        Document cp = null;

        for (SciSummAnnotation annotation : SciSummAnnotationsList) {
            String citance_Number = annotation.getCitance_Number();
            String reference_Article = annotation.getReference_Article();
            String citing_Article = annotation.getCiting_Article();
            String citation_Marker_Offset = annotation.getCitation_Marker_Offset();
            String citation_Marker = annotation.getCitation_Marker();
            ArrayList<String> citation_Offset = annotation.getCitation_Offset();
            String citation_Text = annotation.getCitation_Text();
            if (generateTraining) {
                reference_Offset = annotation.getReference_Offset();
                reference_Text = annotation.getReference_Text();
                discourse_Facet = annotation.getDiscourse_Facet();
            }
            annotator = annotation.getAnnotator();

            rp = RCDocuments.get(reference_Article);
            cp = RCDocuments.get(citing_Article);

            AnnotationSet rpMarkups = rp.getAnnotations("Original markups");
            AnnotationSet cpMarkups = cp.getAnnotations("Original markups");

            String rpContent = rp.getContent().toString();
            String cpContent = cp.getContent().toString();

            AnnotationSet references = rp.getAnnotations("REFERENCES");
            AnnotationSet citations = cp.getAnnotations("CITATIONS");

            AnnotationSet rp_sentences = rpMarkups.get("S");
            AnnotationSet cp_sentences = cpMarkups.get("S");

            FeatureMap cfilter = Factory.newFeatureMap();
            FeatureMap rfilter = Factory.newFeatureMap();

            for (String co : citation_Offset) {
                cfilter.put("sid", co);

                AnnotationSet cselected = cp_sentences.get("S", cfilter);
                Annotation csentence;
                Long cstartS, cendS;
                FeatureMap cfm_anns;
                if (cselected.size() > 0) {
                    csentence = cselected.iterator().next();
                    cstartS = csentence.getStartNode().getOffset();
                    cendS = csentence.getEndNode().getOffset();
                    cfm_anns = Factory.newFeatureMap();
                    cfm_anns.put("id", citance_Number + "_" + reference_Article + "_" + citing_Article + "_" + annotator);
                    cfm_anns.put("Citance_Number", citance_Number);
                    cfm_anns.put("Reference_Article", reference_Article);
                    cfm_anns.put("Citing_Article", citing_Article);
                    cfm_anns.put("Citation_Marker_Offset", citation_Marker_Offset);
                    cfm_anns.put("Citation_Marker", citation_Marker);
                    cfm_anns.put("Citation_Offset", co);
                    //cfm_anns.put("Citation_Text", citation_Text);
                    if (generateTraining) {
                        String facets = "";
                        for (int i = 0; i < discourse_Facet.size(); i++) {
                            facets = facets + discourse_Facet.get(i);
                            if (i < discourse_Facet.size() - 1) {
                                facets = facets + ",";
                            }
                        }
                        cfm_anns.put("Discourse_Facet", facets);
                    }
                    cfm_anns.put("Annotator", annotator);
                    try {
                        citations.add(cstartS, cendS, annotator, cfm_anns);
                    } catch (InvalidOffsetException e) {
                        e.printStackTrace();
                    }
                }
            }
            if (generateTraining) {
                //Annotate The reference
                for (String ro : reference_Offset) {
                    rfilter.put("sid", ro);

                    AnnotationSet rselected = rp_sentences.get("S", rfilter);
                    Annotation rsentence;
                    Long rstartS, rendS;
                    FeatureMap rfm_anns;
                    if (rselected.size() > 0) {
                        rsentence = rselected.iterator().next();
                        rstartS = rsentence.getStartNode().getOffset();
                        rendS = rsentence.getEndNode().getOffset();

                        rfm_anns = Factory.newFeatureMap();
                        rfm_anns.put("id", citance_Number + "_" + reference_Article + "_" + citing_Article + "_" + annotator);
                        rfm_anns.put("Citance_Number", citance_Number);
                        rfm_anns.put("Reference_Article", reference_Article);
                        rfm_anns.put("Citing_Article", citing_Article);
                        rfm_anns.put("Citation_Marker", citation_Marker);
                        rfm_anns.put("Reference_Offset", ro);
                        //rfm_anns.put("reference_Text", reference_Text);
                        String facets = "";
                        for (int i = 0; i < discourse_Facet.size(); i++) {
                            facets = facets + discourse_Facet.get(i);
                            if (i < discourse_Facet.size() - 1) {
                                facets = facets + ",";
                            }
                        }
                        rfm_anns.put("Discourse_Facet", facets);
                        rfm_anns.put("Annotator", annotator);
                        try {
                            references.add(rstartS, rendS, annotator, rfm_anns);
                        } catch (InvalidOffsetException e) {
                            e.printStackTrace();
                        }
                    }
                }
            }

            RCDocuments.put(reference_Article, rp);
            RCDocuments.put(citing_Article, cp);

        }

        processedDocuments = RCDocuments;

        return processedDocuments;
    }

    public static HashMap<String, Document> applyCosineSimilarities
            (HashMap<String, Document> RCDocuments, File rfolder) {
        HashMap<String, Document> processedDocuments = new HashMap<String, Document>();
        HashMap<String, FeatureMap> combinedcpAnnotatorsIDsNormalizedVectors;

        Document rp = RCDocuments.get(rfolder.getName());

        AnnotationSet rpAnalysis = rp.getAnnotations("Analysis");
        AnnotationSet rpNormalizedVectors = rpAnalysis.get("Vector_Norm");

        for (String key : RCDocuments.keySet()) {
            if (!key.equals(rfolder.getName())) {
                Document cp = RCDocuments.get(key);

                AnnotationSet rpSimilarity = rp.getAnnotations("Similarities");
                AnnotationSet cpAnalysis = cp.getAnnotations("Analysis");
                AnnotationSet cpCitMarkups = cp.getAnnotations("CITATIONS");
                AnnotationSet cpNormalizedVectors = cpAnalysis.get("Vector_Norm");
                AnnotationSet rpSentences = rpAnalysis.get("Sentence");

                AnnotationSet rpSentencesNVOffset;

                Annotation rpSentenceNVOffset;

                Long cpStartA, cpEndA, rpStartNV = null, rpEndNV = null, rpMaxStartNVS, rpMAxEndNNVS;
                Iterator cpAnnotatorsIterator = cpCitMarkups.iterator();

                double maxCosineValue = -1;
                int rpMaxNormalizedVectorID = 0;

                combinedcpAnnotatorsIDsNormalizedVectors = combinecpAnnotatorsIDsNormalizedVectors(cpCitMarkups, cpNormalizedVectors);
                for (String id : combinedcpAnnotatorsIDsNormalizedVectors.keySet()) {
                    FeatureMap rpfm_anns = Factory.newFeatureMap();
                    Iterator rpNormalizedVectorsIterator = rpNormalizedVectors.iterator();

                    while (rpNormalizedVectorsIterator.hasNext()) {
                        Annotation rpNormalizedVector = (Annotation) rpNormalizedVectorsIterator.next();
                        rpStartNV = rpNormalizedVector.getStartNode().getOffset();
                        rpEndNV = rpNormalizedVector.getEndNode().getOffset();
                        FeatureMap rpNormalizedVectorfm = rpNormalizedVector.getFeatures();

                        rpSentencesNVOffset = rpSentences.get(rpStartNV);

                        if (rpSentencesNVOffset.size() > 0) {
                            rpSentenceNVOffset = rpSentencesNVOffset.iterator().next();
                            FeatureMap rpSentencefm = rpSentenceNVOffset.getFeatures();
                            rpSentencefm.put("sim_" + id,
                                    summa.scorer.Cosine.cosine1(combinedcpAnnotatorsIDsNormalizedVectors.get(id),
                                            rpNormalizedVectorfm));
                            if (summa.scorer.Cosine.cosine1(combinedcpAnnotatorsIDsNormalizedVectors.get(id),
                                    rpNormalizedVectorfm) > maxCosineValue) {
                                maxCosineValue = summa.scorer.Cosine.cosine1(combinedcpAnnotatorsIDsNormalizedVectors.get(id),
                                        rpNormalizedVectorfm);
                                rpMaxNormalizedVectorID = rpNormalizedVector.getId();
                            }
                        } else {
                            System.out.println("Could not find the Normalized Victor Sentence.");
                        }
                    }

                    Annotation rpMaxNormalizedVector = rpNormalizedVectors.get(rpMaxNormalizedVectorID);
                    rpMaxStartNVS = rpMaxNormalizedVector.getStartNode().getOffset();
                    rpMAxEndNNVS = rpMaxNormalizedVector.getEndNode().getOffset();

                    rpfm_anns.put("MatchCitanceID", id);
                    rpfm_anns.put("MatchSimilarityValue", maxCosineValue);

                    try {
                        rpSimilarity.add(rpMaxStartNVS, rpMAxEndNNVS, "Match", rpfm_anns);
                    } catch (InvalidOffsetException e) {
                        e.printStackTrace();
                    }
                }
                maxCosineValue = -1;
                rpMaxNormalizedVectorID = 0;
            }
        }

        RCDocuments.put(rfolder.getName(), rp);

        processedDocuments = RCDocuments;

        return processedDocuments;
    }

    public static HashMap<String, Document> applyBabelnetCosineSimilarities
            (HashMap<String, Document> RCDocuments, File rfolder) {
        HashMap<String, Document> processedDocuments = new HashMap<String, Document>();
        HashMap<String, FeatureMap> combinedcpAnnotatorsIDsBabelnetNormalizedVectors;

        Document rp = RCDocuments.get(rfolder.getName());

        AnnotationSet rpBabelnet = rp.getAnnotations("Babelnet");
        AnnotationSet rpBabelnetNormalizedVectors = rpBabelnet.get("BNVector_Norm");

        for (String key : RCDocuments.keySet()) {
            if (!key.equals(rfolder.getName())) {
                Document cp = RCDocuments.get(key);

                AnnotationSet rpSimilarity = rp.getAnnotations("BabelnetSimilarities");
                AnnotationSet cpBabelnet = cp.getAnnotations("Babelnet");
                AnnotationSet cpCitMarkups = cp.getAnnotations("CITATIONS");
                AnnotationSet cpBabelnetNormalizedVectors = cpBabelnet.get("BNVector_Norm");
                AnnotationSet rpSentences = rpBabelnet.get("Sentence");

                AnnotationSet rpSentencesNVOffset;

                Annotation rpSentenceNVOffset;

                Long cpStartA, cpEndA, rpStartNV = null, rpEndNV = null, rpMaxStartNVS, rpMAxEndNNVS;
                Iterator cpAnnotatorsIterator = cpCitMarkups.iterator();

                double maxCosineValue = -1;
                int rpMaxBabelnetNormalizedVectorID = 0;

                combinedcpAnnotatorsIDsBabelnetNormalizedVectors = combinecpAnnotatorsIDsNormalizedVectors(cpCitMarkups, cpBabelnetNormalizedVectors);
                for (String id : combinedcpAnnotatorsIDsBabelnetNormalizedVectors.keySet()) {
                    FeatureMap rpfm_anns = Factory.newFeatureMap();
                    Iterator rpBabelnetNormalizedVectorsIterator = rpBabelnetNormalizedVectors.iterator();

                    while (rpBabelnetNormalizedVectorsIterator.hasNext()) {
                        Annotation rpBabelnetNormalizedVector = (Annotation) rpBabelnetNormalizedVectorsIterator.next();
                        rpStartNV = rpBabelnetNormalizedVector.getStartNode().getOffset();
                        rpEndNV = rpBabelnetNormalizedVector.getEndNode().getOffset();
                        FeatureMap rpBabelnetNormalizedVectorfm = rpBabelnetNormalizedVector.getFeatures();

                        rpSentencesNVOffset = rpSentences.get(rpStartNV);

                        if (rpSentencesNVOffset.size() > 0) {
                            rpSentenceNVOffset = rpSentencesNVOffset.iterator().next();
                            FeatureMap rpSentencefm = rpSentenceNVOffset.getFeatures();
                            FeatureMap t = combinedcpAnnotatorsIDsBabelnetNormalizedVectors.get(id);
                            FeatureMap k = rpBabelnetNormalizedVectorfm;
                            double b = summa.scorer.Cosine.cosine1(t, k);

                            rpSentencefm.put("BNsim_" + id,
                                    summa.scorer.Cosine.cosine1(combinedcpAnnotatorsIDsBabelnetNormalizedVectors.get(id),
                                            rpBabelnetNormalizedVectorfm));
                            if (summa.scorer.Cosine.cosine1(combinedcpAnnotatorsIDsBabelnetNormalizedVectors.get(id),
                                    rpBabelnetNormalizedVectorfm) > maxCosineValue) {
                                maxCosineValue = summa.scorer.Cosine.cosine1(combinedcpAnnotatorsIDsBabelnetNormalizedVectors.get(id),
                                        rpBabelnetNormalizedVectorfm);
                                rpMaxBabelnetNormalizedVectorID = rpBabelnetNormalizedVector.getId();
                            }
                        } else {
                            System.out.println("Could not find the Normalized Victor Sentence.");
                        }
                    }

                    Annotation rpMaxBabelnetNormalizedVector = rpBabelnetNormalizedVectors.get(rpMaxBabelnetNormalizedVectorID);
                    rpMaxStartNVS = rpMaxBabelnetNormalizedVector.getStartNode().getOffset();
                    rpMAxEndNNVS = rpMaxBabelnetNormalizedVector.getEndNode().getOffset();

                    rpfm_anns.put("MatchCitanceID", id);
                    rpfm_anns.put("MatchSimilarityValue", maxCosineValue);

                    try {
                        rpSimilarity.add(rpMaxStartNVS, rpMAxEndNNVS, "Match", rpfm_anns);
                    } catch (InvalidOffsetException e) {
                        e.printStackTrace();
                    }
                }
                maxCosineValue = -1;
                rpMaxBabelnetNormalizedVectorID = 0;
            }
        }

        RCDocuments.put(rfolder.getName(), rp);

        processedDocuments = RCDocuments;

        return processedDocuments;
    }

    public static HashMap<String, FeatureMap> combinecpAnnotatorsIDsNormalizedVectors(AnnotationSet
                                                                                              cpAnnotators, AnnotationSet cpNormalizedVectors) {
        HashMap<String, FeatureMap> combinedcpAnnotatorsIDsNormalizedVectors = new HashMap<String, FeatureMap>();
        AnnotationSet cpAnnotatorNormalizedVectors;
        Annotation cpAnnotator;
        Annotation cpAnnotatorNormalizedVector = null;
        FeatureMap cpAnnotatorfm;
        Long cpStartA, cpEndA;

        Iterator cpAnnotatorsIterator = cpAnnotators.iterator();
        while (cpAnnotatorsIterator.hasNext()) {
            cpAnnotator = (Annotation) cpAnnotatorsIterator.next();
            cpStartA = cpAnnotator.getStartNode().getOffset();
            cpEndA = cpAnnotator.getEndNode().getOffset();

            cpAnnotatorfm = cpAnnotator.getFeatures();
            String id = (String) cpAnnotatorfm.get("id");

            if (combinedcpAnnotatorsIDsNormalizedVectors.containsKey(id)) {
                cpAnnotatorNormalizedVectors = cpNormalizedVectors.get(cpStartA);

                if (cpAnnotatorNormalizedVectors.size() > 0) {
                    cpAnnotatorNormalizedVector = cpAnnotatorNormalizedVectors.iterator().next();
                    combinedcpAnnotatorsIDsNormalizedVectors.put(id,
                            combineNormalizedVectors(combinedcpAnnotatorsIDsNormalizedVectors.get(id),
                                    cpAnnotatorNormalizedVector.getFeatures()));
                } else {
                    System.out.println("Could not find the Annotator Normalized Victor.");
                }
            } else {
                cpAnnotatorNormalizedVectors = cpNormalizedVectors.get(cpStartA);
                if (cpAnnotatorNormalizedVectors.size() > 0) {
                    cpAnnotatorNormalizedVector = cpAnnotatorNormalizedVectors.iterator().next();
                    combinedcpAnnotatorsIDsNormalizedVectors.put(id, cpAnnotatorNormalizedVector.getFeatures());
                } else {
                    System.out.println("Could not find the Annotator Normalized Victor.");
                }
            }
        }

        return combinedcpAnnotatorsIDsNormalizedVectors;
    }

    public static FeatureMap combineNormalizedVectors(FeatureMap normalizedVector1, FeatureMap
            normalizedVector2) {
        FeatureMap combineNormalizedVector = Factory.newFeatureMap();
        for (Object key : normalizedVector1.keySet()) {
            if (normalizedVector2.containsKey(key)) {
                combineNormalizedVector.put(key, String.valueOf((new Double((String) normalizedVector1.get(key)) +
                        new Double((String) normalizedVector2.get(key))) / 2.0));
            } else {
                combineNormalizedVector.put(key, String.valueOf((new Double((String) normalizedVector1.get(key)) / 2.0)));
            }
        }

        for (Object key : normalizedVector2.keySet()) {
            if (!normalizedVector1.containsKey(key)) {
                combineNormalizedVector.put(key, String.valueOf((new Double((String) normalizedVector2.get(key)) / 2.0)));
            }
        }
        return combineNormalizedVector;
    }

    public static void exportGATEDocuments(HashMap<String, Document> processedRCDocuments, String
            rfolder, String outputFolder, String extension) {
        PrintWriter pw = null;
        File ref = new File(outputFolder + File.separator + rfolder);

        // attempt to create the directory here
        ref.mkdirs();

        for (String docKey : processedRCDocuments.keySet()) {
            if (ref.exists()) {
                // creating the directory succeeded
                try {
                    pw = new PrintWriter(new OutputStreamWriter(new FileOutputStream(outputFolder + File.separator
                            + rfolder + "/" + docKey + "-" + extension + ".xml"), "UTF-8"));
                } catch (UnsupportedEncodingException e) {
                    e.printStackTrace();
                } catch (FileNotFoundException e) {
                    e.printStackTrace();
                }
                pw.println(processedRCDocuments.get(docKey).toXml());
                pw.flush();
                pw.close();
                Factory.deleteResource(processedRCDocuments.get(docKey));
            } else {
                // creating the directory failed
                System.out.println("failed trying to create the directory");
            }
        }

    }

    public static Pair<gate.Document, Integer> annotateBabelnetAnnotations(gate.Document document, Integer queryCounter) {

        URL url;
        InputStream inputStream = null;
        DataInputStream dataInputStream;
        String s;

        AnnotationSet documentSentences = document.getAnnotations("Original markups").get("S");
        AnnotationSet documentBabelnet = document.getAnnotations("Babelnet");
        documentBabelnet.clear();

        List annotationsList = new ArrayList((AnnotationSet) documentSentences);
        Collections.sort(annotationsList, new OffsetComparator());

        Long start = 0l, end = 0l, size = 0l;

        for (Object documentSentenceAnnotation : annotationsList) {
            try {
                Annotation documentSentence = (Annotation) documentSentenceAnnotation;
                if (((documentSentence.getEndNode().getOffset() - documentSentence.getStartNode().getOffset()) + size <= 1700) &&
                        ((documentSentence.getEndNode().getOffset() - documentSentence.getStartNode().getOffset()) <= 1000) &&
                        (!documentSentence.equals(annotationsList.get(annotationsList.size() - 1)))) {
                    end = documentSentence.getEndNode().getOffset();
                    size = end - start;
                } else {
                    if (documentSentence.equals(annotationsList.get(annotationsList.size() - 1))) {
                        end = documentSentence.getEndNode().getOffset();
                    }
                    if (start == end) {
                        continue;
                    }

                    url = new URL("http://babelfy.io/v1/disambiguate?text=" + URLEncoder.encode(document.getContent().getContent(start, end).toString(), "UTF-8") + "&lang=EN&key=3567f537-c43b-4176-ac95-6f10dc7becef");
                    inputStream = url.openStream();         // throws an IOException
                    dataInputStream = new DataInputStream(new BufferedInputStream(inputStream));
                    StringBuffer stringBuffer = new StringBuffer();
                    while ((s = dataInputStream.readLine()) != null) {
                        stringBuffer.append(s);
                    }
                    System.out.println("Processing context: " + document.getContent().getContent(start, end).toString());
                    //System.out.println("Json Response: " + stringBuffer.toString());

                    queryCounter++;

                    ObjectMapper mapper = new ObjectMapper();
                    BabelnetSynset[] synsets = mapper.readValue(stringBuffer.toString(), BabelnetSynset[].class);

                    for (BabelnetSynset synset : synsets) {
                        BabelnetSynset.TokenFragment tokenFragment = synset.getTokenFragment();
                        BabelnetSynset.CharFragment charFragment = synset.getCharFragment();

                        Long tokenFragmentStart = Long.parseLong(tokenFragment.getStart());
                        Long tokenFragmentEnd = Long.parseLong(tokenFragment.getEnd());
                        Long charFragmentStart = start + Long.parseLong(charFragment.getStart());
                        Long charFragmentEnd = start + Long.parseLong(charFragment.getEnd()) + 1L;

                        FeatureMap fm = Factory.newFeatureMap();
                        fm.put("babelnetURL", synset.getBabelNetURL());
                        fm.put("coherenceScore", synset.getCoherenceScore());
                        fm.put("dbpediaURL", synset.getDBpediaURL());
                        fm.put("globalScore", synset.getGlobalScore());
                        fm.put("numTokens", (tokenFragmentEnd - tokenFragmentStart) + 1);
                        fm.put("score", synset.getScore());
                        fm.put("source", synset.getSource());
                        fm.put("synsetID", synset.getBabelSynsetID());

                        documentBabelnet.add(charFragmentStart, charFragmentEnd, "Entity", fm);
                    }
                    if ((documentSentence.getEndNode().getOffset() - documentSentence.getStartNode().getOffset()) <= 1000) {
                        start = documentSentence.getStartNode().getOffset();
                        end = documentSentence.getEndNode().getOffset();
                        size = end - start;
                    } else {
                        start = documentSentence.getEndNode().getOffset() + 1;
                        end = documentSentence.getEndNode().getOffset() + 1;
                        size = 0l;
                    }

                }

            } catch (InvalidOffsetException e) {
                e.printStackTrace();
            } catch (MalformedURLException e) {
                e.printStackTrace();
            } catch (UnsupportedEncodingException e) {
                e.printStackTrace();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return Pair.of(document, queryCounter);
    }

    public static gate.Document annotateWord2VecAnnotations(gate.Document document, Word2Vec word2Vec, String annotationSetName) {
        Iterator tokensIterator = document.getAnnotations("Analysis").get("Token").iterator();
        AnnotationSet annotationSet = document.getAnnotations("Word2Vec");

        while (tokensIterator.hasNext()) {
            Annotation token = (Annotation) tokensIterator.next();
            if (token.getFeatures().get("kind").toString().equals("word")) {

                if (word2Vec.hasWord(token.getFeatures().get("string").toString().toLowerCase())) {
                    double[] wordVector = word2Vec.getWordVector(token.getFeatures().get("string").toString().toLowerCase());
                    if (wordVector != null) {
                        FeatureMap fm = Factory.newFeatureMap();
                        StringBuilder vector = new StringBuilder();
                        for (int i = 0; i < wordVector.length; i++) {
                            vector.append(wordVector[i]);
                            vector.append(" ");
                        }
                        fm.put(token.getFeatures().get("string").toString().toLowerCase(), vector.toString().trim());
                        try {
                            annotationSet.add(token.getStartNode().getOffset(), token.getEndNode().getOffset(), annotationSetName, fm);
                        } catch (InvalidOffsetException e) {
                            e.printStackTrace();
                        }
                    }
                }
            }
        }

        return document;
    }

    public static Document fillDocumentMissingLemmas(Document document) {
        for (Annotation annotation : document.getAnnotations("Analysis").get("Token")) {
            FeatureMap fm = annotation.getFeatures();
            if (!fm.containsKey("lemma")) {
                if (fm.containsKey("string")) {
                    fm.put("lemma", fm.get("string"));
                } else {
                    try {
                        String value = String.valueOf(document.getContent().getContent(annotation.getStartNode().getOffset(), annotation.getEndNode().getOffset()));
                        fm.put("lemma", value.toLowerCase());
                    } catch (InvalidOffsetException e) {
                        e.printStackTrace();
                    }
                }
            }
        }
        return document;
    }

    public static Document fillDocumentBabelNetKind(Document document) {
        for (Annotation annotation : document.getAnnotations("Babelnet").get("Entity")) {
            FeatureMap fm = annotation.getFeatures();
            if (!fm.containsKey("kind")) {
                fm.put("kind", "entity");
            }
        }
        return document;
    }

    public static Document fillDocumentMissingPOS(Document document) {
        for (Annotation annotation : document.getAnnotations("Analysis").get("Token")) {
            FeatureMap fm = annotation.getFeatures();
            if (!fm.containsKey("category")) {
                fm.put("category", "-LRB-");
            }
        }
        return document;
    }

    public static FeatureSet<TrainingExample, DocumentCtx> generateMatchFeatureSet() {

        FeatureSet<TrainingExample, DocumentCtx> featSet = new FeatureSet<TrainingExample, DocumentCtx>();

        // Adding document identifier
        try {
            featSet.addFeature(new StringW<TrainingExample, DocumentCtx>("ID", new ID()));

            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("SENTENCE_POSITION", new SentencePosition("sid")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("SENTENCE_SECTION_POSITION", new SentencePosition("ssid")));

            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("FACET_AIM", new SectionTitleFacet(new String[]{"aim", "objective", "purpose"})));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("FACET_HYPOTHESIS", new SectionTitleFacet(new String[]{"hypothesis", "possibility", "theory"})));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("FACET_IMPLICATION", new SectionTitleFacet(new String[]{"implication", "deduction", "entailment"})));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("FACET_METHOD", new SectionTitleFacet(new String[]{"method", "approach"})));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("FACET_RESULT", new SectionTitleFacet(new String[]{"result", "solution", "outcome", "answer", "evaluation"})));

            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("JIANGCONRATH_SIMILARITY", new WordNetSimilarity(true, "jiangconrath")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("LCH_SIMILARITY", new WordNetSimilarity(true, "lch")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("LESK_SIMILARITY", new WordNetSimilarity(true, "lesk")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("LIN_SIMILARITY", new WordNetSimilarity(true, "lin")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("PATH_SIMILARITY", new WordNetSimilarity(true, "path")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RESNIK_SIMILARITY", new WordNetSimilarity(true, "resnik")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("WUP_SIMILARITY", new WordNetSimilarity(true, "wup")));

            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("COSINE_SIMILARITY", new CosineSimilarity("LEMMA")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("BABELNET_COSINE_SIMILARITY", new CosineSimilarity("BABELNET")));

            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("Jaccard", new Jaccard(8, 3)));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("IdfWeightedJaccard", new IdfWeightedJaccard(8, 3)));

            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("PROBABILITY_APPROACH", new DrInventorFacetProbability("PROB_DRI_Approach")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("PROBABILITY_BACKGROUND", new DrInventorFacetProbability("PROB_DRI_Background")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("PROBABILITY_CHALLENGE", new DrInventorFacetProbability("PROB_DRI_Challenge")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("PROBABILITY_FUTUREWORK", new DrInventorFacetProbability("PROB_DRI_FutureWork")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("PROBABILITY_OUTCOME", new DrInventorFacetProbability("PROB_DRI_Outcome")));

            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CP_CITMARKER_COUNT", new CitationMarkerCount(true, "CP")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RP_CITMARKER_COUNT", new CitationMarkerCount(true, "RP")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CITMARKER_COUNT", new CitationMarkerCount(true, "BOTH")));

            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CP_CAUSEAFFECT_EXISTANCE", new CauseAffectExistance(true, "CP")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RP_CAUSEAFFECT_EXISTANCE", new CauseAffectExistance(true, "RP")));

            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CP_COREFCHAINS_COUNT", new CoRefChainsCount(true, "CP")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RP_COREFCHAINS_COUNT", new CoRefChainsCount(true, "RP")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("COREFCHAINS_COUNT", new CoRefChainsCount(true, "BOTH")));

            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZRESEARCHMT_PROP", new GazProbability("BOTH", "research")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZRESEARCHMT_PROP", new GazProbability("RP", "research")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZRESEARCHMT_PROP", new GazProbability("CP", "research")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZARGUMENTATIONMT_PROP", new GazProbability("BOTH", "argumentation")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZARGUMENTATIONMT_PROP", new GazProbability("RP", "argumentation")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZARGUMENTATIONMT_PROP", new GazProbability("CP", "argumentation")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZAWAREMT_PROP", new GazProbability("BOTH", "aware")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZAWAREMT_PROP", new GazProbability("RP", "aware")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZAWAREMT_PROP", new GazProbability("CP", "aware")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZUSEMT_PROP", new GazProbability("BOTH", "use")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZUSEMT_PROP", new GazProbability("RP", "use")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZUSEMT_PROP", new GazProbability("CP", "use")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZPROBLEMMT_PROP", new GazProbability("BOTH", "problem")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZPROBLEMMT_PROP", new GazProbability("RP", "problem")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZPROBLEMMT_PROP", new GazProbability("CP", "problem")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZSOLUTIONMT_PROP", new GazProbability("BOTH", "solution")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZSOLUTIONMT_PROP", new GazProbability("RP", "solution")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZSOLUTIONMT_PROP", new GazProbability("CP", "solution")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZBETTERSOLUTIONMT_PROP", new GazProbability("BOTH", "better_solution")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZBETTERSOLUTIONMT_PROP", new GazProbability("RP", "better_solution")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZBETTERSOLUTIONMT_PROP", new GazProbability("CP", "better_solution")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZTEXTSTRUCTUREMT_PROP", new GazProbability("BOTH", "textstructure")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZTEXTSTRUCTUREMT_PROP", new GazProbability("RP", "textstructure")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZTEXTSTRUCTUREMT_PROP", new GazProbability("CP", "textstructure")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZINTRESTMT_PROP", new GazProbability("BOTH", "interest")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZINTRESTMT_PROP", new GazProbability("RP", "interest")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZINTRESTMT_PROP", new GazProbability("CP", "interest")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZCONTINUEMT_PROP", new GazProbability("BOTH", "continue")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZCONTINUEMT_PROP", new GazProbability("RP", "continue")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZCONTINUEMT_PROP", new GazProbability("CP", "continue")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZFUTUREINTERESTMT_PROP", new GazProbability("BOTH", "future_interest")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZFUTUREINTERESTMT_PROP", new GazProbability("RP", "future_interest")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZFUTUREINTERESTMT_PROP", new GazProbability("CP", "future_interest")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZNEEDMT_PROP", new GazProbability("BOTH", "need")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZNEEDMT_PROP", new GazProbability("RP", "need")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZNEEDMT_PROP", new GazProbability("CP", "need")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZAFFECTMT_PROP", new GazProbability("BOTH", "affect")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZAFFECTMT_PROP", new GazProbability("RP", "affect")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZAFFECTMT_PROP", new GazProbability("CP", "affect")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZPRESENTATIONMT_PROP", new GazProbability("BOTH", "presentation")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZPRESENTATIONMT_PROP", new GazProbability("RP", "presentation")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZPRESENTATIONMT_PROP", new GazProbability("CP", "presentation")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZCONTRASTMT_PROP", new GazProbability("BOTH", "contrast")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZCONTRASTMT_PROP", new GazProbability("RP", "contrast")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZCONTRASTMT_PROP", new GazProbability("CP", "contrast")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZCHANGEMT_PROP", new GazProbability("BOTH", "change")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZCHANGEMT_PROP", new GazProbability("RP", "change")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZCHANGEMT_PROP", new GazProbability("CP", "change")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZCOMPARISONMT_PROP", new GazProbability("BOTH", "comparison")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZCOMPARISONMT_PROP", new GazProbability("RP", "comparison")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZCOMPARISONMT_PROP", new GazProbability("CP", "comparison")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZSIMILARMT_PROP", new GazProbability("BOTH", "similar")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZSIMILARMT_PROP", new GazProbability("RP", "similar")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZSIMILARMT_PROP", new GazProbability("CP", "similar")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZCOMPARISONADJMT_PROP", new GazProbability("BOTH", "comparison_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZCOMPARISONADJMT_PROP", new GazProbability("RP", "comparison_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZCOMPARISONADJMT_PROP", new GazProbability("CP", "comparison_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZFUTUREADJMT_PROP", new GazProbability("BOTH", "future_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZFUTUREADJMT_PROP", new GazProbability("RP", "future_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZFUTUREADJMT_PROP", new GazProbability("CP", "future_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZINTERESTNOUNMT_PROP", new GazProbability("BOTH", "interest_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZINTERESTNOUNMT_PROP", new GazProbability("RP", "interest_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZINTERESTNOUNMT_PROP", new GazProbability("CP", "interest_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZQUESTIONNOUNMT_PROP", new GazProbability("BOTH", "question_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZQUESTIONNOUNMT_PROP", new GazProbability("RP", "question_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZQUESTIONNOUNMT_PROP", new GazProbability("CP", "question_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZAWAREADJMT_PROP", new GazProbability("BOTH", "aware_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZAWAREADJMT_PROP", new GazProbability("RP", "aware_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZAWAREADJMT_PROP", new GazProbability("CP", "aware_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZARGUMENTATIONNOUNMT_PROP", new GazProbability("BOTH", "argumentation_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZARGUMENTATIONNOUNMT_PROP", new GazProbability("RP", "argumentation_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZARGUMENTATIONNOUNMT_PROP", new GazProbability("CP", "argumentation_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZSIMILARNOUNMT_PROP", new GazProbability("BOTH", "similar_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZSIMILARNOUNMT_PROP", new GazProbability("RP", "similar_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZSIMILARNOUNMT_PROP", new GazProbability("CP", "similar_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZEARLIERADJMT_PROP", new GazProbability("BOTH", "earlier_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZEARLIERADJMT_PROP", new GazProbability("RP", "earlier_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZEARLIERADJMT_PROP", new GazProbability("CP", "earlier_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZRESEARCHADJMT_PROP", new GazProbability("BOTH", "research_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZRESEARCHADJMT_PROP", new GazProbability("RP", "research_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZRESEARCHADJMT_PROP", new GazProbability("CP", "research_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZNEEDADJMT_PROP", new GazProbability("BOTH", "need_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZNEEDADJMT_PROP", new GazProbability("RP", "need_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZNEEDADJMT_PROP", new GazProbability("CP", "need_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZREFERENTIALMT_PROP", new GazProbability("BOTH", "referential")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZREFERENTIALMT_PROP", new GazProbability("RP", "referential")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZREFERENTIALMT_PROP", new GazProbability("CP", "referential")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZQUESTIONMT_PROP", new GazProbability("BOTH", "question")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZQUESTIONMT_PROP", new GazProbability("RP", "question")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZQUESTIONMT_PROP", new GazProbability("CP", "question")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZWORKNOUNMT_PROP", new GazProbability("BOTH", "work_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZWORKNOUNMT_PROP", new GazProbability("RP", "work_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZWORKNOUNMT_PROP", new GazProbability("CP", "work_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZCHANGEADJMT_PROP", new GazProbability("BOTH", "change_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZCHANGEADJMT_PROP", new GazProbability("RP", "change_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZCHANGEADJMT_PROP", new GazProbability("CP", "change_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZDISCIPLINEMT_PROP", new GazProbability("BOTH", "discipline")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZDISCIPLINEMT_PROP", new GazProbability("RP", "discipline")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZDISCIPLINEMT_PROP", new GazProbability("CP", "discipline")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZGIVENMT_PROP", new GazProbability("BOTH", "given")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZGIVENMT_PROP", new GazProbability("RP", "given")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZGIVENMT_PROP", new GazProbability("CP", "given")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZBADADJMT_PROP", new GazProbability("BOTH", "bad_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZBADADJMT_PROP", new GazProbability("RP", "bad_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZBADADJMT_PROP", new GazProbability("CP", "bad_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZCONTRASTNOUNMT_PROP", new GazProbability("BOTH", "contrast_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZCONTRASTNOUNMT_PROP", new GazProbability("RP", "contrast_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZCONTRASTNOUNMT_PROP", new GazProbability("CP", "contrast_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZNEEDNOUNMT_PROP", new GazProbability("BOTH", "need_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZNEEDNOUNMT_PROP", new GazProbability("RP", "need_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZNEEDNOUNMT_PROP", new GazProbability("CP", "need_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZAIMNOUNMT_PROP", new GazProbability("BOTH", "aim_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZAIMNOUNMT_PROP", new GazProbability("RP", "aim_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZAIMNOUNMT_PROP", new GazProbability("CP", "aim_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZCONTRASTADJMT_PROP", new GazProbability("BOTH", "contrast_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZCONTRASTADJMT_PROP", new GazProbability("RP", "contrast_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZCONTRASTADJMT_PROP", new GazProbability("CP", "contrast_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZSOLUTIONNOUNMT_PROP", new GazProbability("BOTH", "solution_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZSOLUTIONNOUNMT_PROP", new GazProbability("RP", "solution_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZSOLUTIONNOUNMT_PROP", new GazProbability("CP", "solution_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZTRADITIONNOUNMT_PROP", new GazProbability("BOTH", "tradition_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZTRADITIONNOUNMT_PROP", new GazProbability("RP", "tradition_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZTRADITIONNOUNMT_PROP", new GazProbability("CP", "tradition_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZFIRSTPRONMT_PROP", new GazProbability("BOTH", "first_pron")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZFIRSTPRONMT_PROP", new GazProbability("RP", "first_pron")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZFIRSTPRONMT_PROP", new GazProbability("CP", "first_pron")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZPROFESSIONALSMT_PROP", new GazProbability("BOTH", "professionals")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZPROFESSIONALSMT_PROP", new GazProbability("RP", "professionals")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZPROFESSIONALSMT_PROP", new GazProbability("CP", "professionals")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZPROBLEMNOUNMT_PROP", new GazProbability("BOTH", "problem_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZPROBLEMNOUNMT_PROP", new GazProbability("RP", "problem_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZPROBLEMNOUNMT_PROP", new GazProbability("CP", "problem_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZNEGATIONMT_PROP", new GazProbability("BOTH", "negation")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZNEGATIONMT_PROP", new GazProbability("RP", "negation")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZNEGATIONMT_PROP", new GazProbability("CP", "negation")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZTEXTNOUNMT_PROP", new GazProbability("BOTH", "text_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZTEXTNOUNMT_PROP", new GazProbability("RP", "text_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZTEXTNOUNMT_PROP", new GazProbability("CP", "text_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZPROBLEMADJMT_PROP", new GazProbability("BOTH", "problem_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZPROBLEMADJMT_PROP", new GazProbability("RP", "problem_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZPROBLEMADJMT_PROP", new GazProbability("CP", "problem_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZTHIRDPRONMT_PROP", new GazProbability("BOTH", "third_pron")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZTHIRDPRONMT_PROP", new GazProbability("RP", "third_pron")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZTHIRDPRONMT_PROP", new GazProbability("CP", "third_pron")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZTRADITIONADJMT_PROP", new GazProbability("BOTH", "tradition_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZTRADITIONADJMT_PROP", new GazProbability("RP", "tradition_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZTRADITIONADJMT_PROP", new GazProbability("CP", "tradition_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZPRESENTATIONNOUNMT_PROP", new GazProbability("BOTH", "presentation_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZPRESENTATIONNOUNMT_PROP", new GazProbability("RP", "presentation_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZPRESENTATIONNOUNMT_PROP", new GazProbability("CP", "presentation_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZRESEARCHNOUNMT_PROP", new GazProbability("BOTH", "research_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZRESEARCHNOUNMT_PROP", new GazProbability("RP", "research_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZRESEARCHNOUNMT_PROP", new GazProbability("CP", "research_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZMAINADJMT_PROP", new GazProbability("BOTH", "main_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZMAINADJMT_PROP", new GazProbability("RP", "main_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZMAINADJMT_PROP", new GazProbability("CP", "main_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZREFLEXSIVEMT_PROP", new GazProbability("BOTH", "reflexive")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZREFLEXSIVEMT_PROP", new GazProbability("RP", "reflexive")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZREFLEXSIVEMT_PROP", new GazProbability("CP", "reflexive")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZNEDADJMT_PROP", new GazProbability("BOTH", "ned_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZNEDADJMT_PROP", new GazProbability("RP", "ned_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZNEDADJMT_PROP", new GazProbability("CP", "ned_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZMANYMT_PROP", new GazProbability("BOTH", "many")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZMANYMT_PROP", new GazProbability("RP", "many")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZMANYMT_PROP", new GazProbability("CP", "many")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZCOMPARISONNOUNMT_PROP", new GazProbability("BOTH", "comparison_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZCOMPARISONNOUNMT_PROP", new GazProbability("RP", "comparison_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZCOMPARISONNOUNMT_PROP", new GazProbability("CP", "comparison_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZGOODADJMT_PROP", new GazProbability("BOTH", "good_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZGOODADJMT_PROP", new GazProbability("RP", "good_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZGOODADJMT_PROP", new GazProbability("CP", "good_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZCHANGENOUNMT_PROP", new GazProbability("BOTH", "change_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZCHANGENOUNMT_PROP", new GazProbability("RP", "change_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZCHANGENOUNMT_PROP", new GazProbability("CP", "change_noun")));

            Set<String> matchClassValues = new HashSet<String>();

            matchClassValues.add("MATCH");
            matchClassValues.add("NO_MATCH");

            // Class feature (lasts)
            featSet.addFeature(new NominalW<TrainingExample, DocumentCtx>("class", matchClassValues, new ClassGetter(true)));

        } catch (FeatureException e) {
            System.out.println("Error instantiating feature generation template.");
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }

        return featSet;
    }

    public static FeatureSet<TrainingExample, DocumentCtx> generateFacetFeatureSet(String type) {
        FeatureSet<TrainingExample, DocumentCtx> featSet = new FeatureSet<TrainingExample, DocumentCtx>();

        // Adding document identifier
        try {
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("SENTENCE_POSITION", new SentencePosition("sid")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("SENTENCE_SECTION_POSITION", new SentencePosition("ssid")));

            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("FACET_AIM", new SectionTitleFacet(new String[]{"aim", "objective", "purpose"})));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("FACET_HYPOTHESIS", new SectionTitleFacet(new String[]{"hypothesis", "possibility", "theory"})));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("FACET_IMPLICATION", new SectionTitleFacet(new String[]{"implication", "deduction", "entailment"})));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("FACET_METHOD", new SectionTitleFacet(new String[]{"method", "approach"})));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("FACET_RESULT", new SectionTitleFacet(new String[]{"result", "solution", "outcome", "answer", "evaluation"})));

            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("JIANGCONRATH_SIMILARITY", new WordNetSimilarity(true, "jiangconrath")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("LCH_SIMILARITY", new WordNetSimilarity(true, "lch")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("LESK_SIMILARITY", new WordNetSimilarity(true, "lesk")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("LIN_SIMILARITY", new WordNetSimilarity(true, "lin")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("PATH_SIMILARITY", new WordNetSimilarity(true, "path")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RESNIK_SIMILARITY", new WordNetSimilarity(true, "resnik")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("WUP_SIMILARITY", new WordNetSimilarity(true, "wup")));

            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("COSINE_SIMILARITY", new CosineSimilarity("LEMMA")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("BABELNET_COSINE_SIMILARITY", new CosineSimilarity("BABELNET")));

            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("Jaccard", new Jaccard(8, 3)));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("IdfWeightedJaccard", new IdfWeightedJaccard(8, 3)));

            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("PROBABILITY_APPROACH", new DrInventorFacetProbability("PROB_DRI_Approach")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("PROBABILITY_BACKGROUND", new DrInventorFacetProbability("PROB_DRI_Background")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("PROBABILITY_CHALLENGE", new DrInventorFacetProbability("PROB_DRI_Challenge")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("PROBABILITY_FUTUREWORK", new DrInventorFacetProbability("PROB_DRI_FutureWork")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("PROBABILITY_OUTCOME", new DrInventorFacetProbability("PROB_DRI_Outcome")));

            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CP_CITMARKER_COUNT", new CitationMarkerCount(true, "CP")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RP_CITMARKER_COUNT", new CitationMarkerCount(true, "RP")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CITMARKER_COUNT", new CitationMarkerCount(true, "BOTH")));

            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CP_CAUSEAFFECT_EXISTANCE", new CauseAffectExistance(true, "CP")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RP_CAUSEAFFECT_EXISTANCE", new CauseAffectExistance(true, "RP")));

            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CP_COREFCHAINS_COUNT", new CoRefChainsCount(true, "CP")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RP_COREFCHAINS_COUNT", new CoRefChainsCount(true, "RP")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("COREFCHAINS_COUNT", new CoRefChainsCount(true, "BOTH")));

            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZRESEARCHMT_PROP", new GazProbability("BOTH", "research")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZRESEARCHMT_PROP", new GazProbability("RP", "research")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZRESEARCHMT_PROP", new GazProbability("CP", "research")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZARGUMENTATIONMT_PROP", new GazProbability("BOTH", "argumentation")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZARGUMENTATIONMT_PROP", new GazProbability("RP", "argumentation")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZARGUMENTATIONMT_PROP", new GazProbability("CP", "argumentation")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZAWAREMT_PROP", new GazProbability("BOTH", "aware")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZAWAREMT_PROP", new GazProbability("RP", "aware")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZAWAREMT_PROP", new GazProbability("CP", "aware")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZUSEMT_PROP", new GazProbability("BOTH", "use")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZUSEMT_PROP", new GazProbability("RP", "use")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZUSEMT_PROP", new GazProbability("CP", "use")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZPROBLEMMT_PROP", new GazProbability("BOTH", "problem")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZPROBLEMMT_PROP", new GazProbability("RP", "problem")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZPROBLEMMT_PROP", new GazProbability("CP", "problem")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZSOLUTIONMT_PROP", new GazProbability("BOTH", "solution")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZSOLUTIONMT_PROP", new GazProbability("RP", "solution")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZSOLUTIONMT_PROP", new GazProbability("CP", "solution")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZBETTERSOLUTIONMT_PROP", new GazProbability("BOTH", "better_solution")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZBETTERSOLUTIONMT_PROP", new GazProbability("RP", "better_solution")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZBETTERSOLUTIONMT_PROP", new GazProbability("CP", "better_solution")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZTEXTSTRUCTUREMT_PROP", new GazProbability("BOTH", "textstructure")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZTEXTSTRUCTUREMT_PROP", new GazProbability("RP", "textstructure")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZTEXTSTRUCTUREMT_PROP", new GazProbability("CP", "textstructure")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZINTRESTMT_PROP", new GazProbability("BOTH", "interest")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZINTRESTMT_PROP", new GazProbability("RP", "interest")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZINTRESTMT_PROP", new GazProbability("CP", "interest")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZCONTINUEMT_PROP", new GazProbability("BOTH", "continue")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZCONTINUEMT_PROP", new GazProbability("RP", "continue")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZCONTINUEMT_PROP", new GazProbability("CP", "continue")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZFUTUREINTERESTMT_PROP", new GazProbability("BOTH", "future_interest")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZFUTUREINTERESTMT_PROP", new GazProbability("RP", "future_interest")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZFUTUREINTERESTMT_PROP", new GazProbability("CP", "future_interest")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZNEEDMT_PROP", new GazProbability("BOTH", "need")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZNEEDMT_PROP", new GazProbability("RP", "need")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZNEEDMT_PROP", new GazProbability("CP", "need")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZAFFECTMT_PROP", new GazProbability("BOTH", "affect")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZAFFECTMT_PROP", new GazProbability("RP", "affect")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZAFFECTMT_PROP", new GazProbability("CP", "affect")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZPRESENTATIONMT_PROP", new GazProbability("BOTH", "presentation")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZPRESENTATIONMT_PROP", new GazProbability("RP", "presentation")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZPRESENTATIONMT_PROP", new GazProbability("CP", "presentation")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZCONTRASTMT_PROP", new GazProbability("BOTH", "contrast")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZCONTRASTMT_PROP", new GazProbability("RP", "contrast")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZCONTRASTMT_PROP", new GazProbability("CP", "contrast")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZCHANGEMT_PROP", new GazProbability("BOTH", "change")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZCHANGEMT_PROP", new GazProbability("RP", "change")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZCHANGEMT_PROP", new GazProbability("CP", "change")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZCOMPARISONMT_PROP", new GazProbability("BOTH", "comparison")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZCOMPARISONMT_PROP", new GazProbability("RP", "comparison")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZCOMPARISONMT_PROP", new GazProbability("CP", "comparison")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZSIMILARMT_PROP", new GazProbability("BOTH", "similar")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZSIMILARMT_PROP", new GazProbability("RP", "similar")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZSIMILARMT_PROP", new GazProbability("CP", "similar")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZCOMPARISONADJMT_PROP", new GazProbability("BOTH", "comparison_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZCOMPARISONADJMT_PROP", new GazProbability("RP", "comparison_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZCOMPARISONADJMT_PROP", new GazProbability("CP", "comparison_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZFUTUREADJMT_PROP", new GazProbability("BOTH", "future_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZFUTUREADJMT_PROP", new GazProbability("RP", "future_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZFUTUREADJMT_PROP", new GazProbability("CP", "future_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZINTERESTNOUNMT_PROP", new GazProbability("BOTH", "interest_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZINTERESTNOUNMT_PROP", new GazProbability("RP", "interest_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZINTERESTNOUNMT_PROP", new GazProbability("CP", "interest_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZQUESTIONNOUNMT_PROP", new GazProbability("BOTH", "question_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZQUESTIONNOUNMT_PROP", new GazProbability("RP", "question_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZQUESTIONNOUNMT_PROP", new GazProbability("CP", "question_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZAWAREADJMT_PROP", new GazProbability("BOTH", "aware_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZAWAREADJMT_PROP", new GazProbability("RP", "aware_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZAWAREADJMT_PROP", new GazProbability("CP", "aware_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZARGUMENTATIONNOUNMT_PROP", new GazProbability("BOTH", "argumentation_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZARGUMENTATIONNOUNMT_PROP", new GazProbability("RP", "argumentation_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZARGUMENTATIONNOUNMT_PROP", new GazProbability("CP", "argumentation_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZSIMILARNOUNMT_PROP", new GazProbability("BOTH", "similar_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZSIMILARNOUNMT_PROP", new GazProbability("RP", "similar_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZSIMILARNOUNMT_PROP", new GazProbability("CP", "similar_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZEARLIERADJMT_PROP", new GazProbability("BOTH", "earlier_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZEARLIERADJMT_PROP", new GazProbability("RP", "earlier_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZEARLIERADJMT_PROP", new GazProbability("CP", "earlier_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZRESEARCHADJMT_PROP", new GazProbability("BOTH", "research_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZRESEARCHADJMT_PROP", new GazProbability("RP", "research_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZRESEARCHADJMT_PROP", new GazProbability("CP", "research_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZNEEDADJMT_PROP", new GazProbability("BOTH", "need_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZNEEDADJMT_PROP", new GazProbability("RP", "need_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZNEEDADJMT_PROP", new GazProbability("CP", "need_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZREFERENTIALMT_PROP", new GazProbability("BOTH", "referential")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZREFERENTIALMT_PROP", new GazProbability("RP", "referential")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZREFERENTIALMT_PROP", new GazProbability("CP", "referential")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZQUESTIONMT_PROP", new GazProbability("BOTH", "question")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZQUESTIONMT_PROP", new GazProbability("RP", "question")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZQUESTIONMT_PROP", new GazProbability("CP", "question")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZWORKNOUNMT_PROP", new GazProbability("BOTH", "work_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZWORKNOUNMT_PROP", new GazProbability("RP", "work_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZWORKNOUNMT_PROP", new GazProbability("CP", "work_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZCHANGEADJMT_PROP", new GazProbability("BOTH", "change_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZCHANGEADJMT_PROP", new GazProbability("RP", "change_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZCHANGEADJMT_PROP", new GazProbability("CP", "change_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZDISCIPLINEMT_PROP", new GazProbability("BOTH", "discipline")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZDISCIPLINEMT_PROP", new GazProbability("RP", "discipline")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZDISCIPLINEMT_PROP", new GazProbability("CP", "discipline")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZGIVENMT_PROP", new GazProbability("BOTH", "given")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZGIVENMT_PROP", new GazProbability("RP", "given")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZGIVENMT_PROP", new GazProbability("CP", "given")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZBADADJMT_PROP", new GazProbability("BOTH", "bad_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZBADADJMT_PROP", new GazProbability("RP", "bad_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZBADADJMT_PROP", new GazProbability("CP", "bad_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZCONTRASTNOUNMT_PROP", new GazProbability("BOTH", "contrast_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZCONTRASTNOUNMT_PROP", new GazProbability("RP", "contrast_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZCONTRASTNOUNMT_PROP", new GazProbability("CP", "contrast_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZNEEDNOUNMT_PROP", new GazProbability("BOTH", "need_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZNEEDNOUNMT_PROP", new GazProbability("RP", "need_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZNEEDNOUNMT_PROP", new GazProbability("CP", "need_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZAIMNOUNMT_PROP", new GazProbability("BOTH", "aim_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZAIMNOUNMT_PROP", new GazProbability("RP", "aim_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZAIMNOUNMT_PROP", new GazProbability("CP", "aim_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZCONTRASTADJMT_PROP", new GazProbability("BOTH", "contrast_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZCONTRASTADJMT_PROP", new GazProbability("RP", "contrast_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZCONTRASTADJMT_PROP", new GazProbability("CP", "contrast_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZSOLUTIONNOUNMT_PROP", new GazProbability("BOTH", "solution_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZSOLUTIONNOUNMT_PROP", new GazProbability("RP", "solution_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZSOLUTIONNOUNMT_PROP", new GazProbability("CP", "solution_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZTRADITIONNOUNMT_PROP", new GazProbability("BOTH", "tradition_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZTRADITIONNOUNMT_PROP", new GazProbability("RP", "tradition_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZTRADITIONNOUNMT_PROP", new GazProbability("CP", "tradition_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZFIRSTPRONMT_PROP", new GazProbability("BOTH", "first_pron")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZFIRSTPRONMT_PROP", new GazProbability("RP", "first_pron")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZFIRSTPRONMT_PROP", new GazProbability("CP", "first_pron")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZPROFESSIONALSMT_PROP", new GazProbability("BOTH", "professionals")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZPROFESSIONALSMT_PROP", new GazProbability("RP", "professionals")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZPROFESSIONALSMT_PROP", new GazProbability("CP", "professionals")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZPROBLEMNOUNMT_PROP", new GazProbability("BOTH", "problem_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZPROBLEMNOUNMT_PROP", new GazProbability("RP", "problem_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZPROBLEMNOUNMT_PROP", new GazProbability("CP", "problem_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZNEGATIONMT_PROP", new GazProbability("BOTH", "negation")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZNEGATIONMT_PROP", new GazProbability("RP", "negation")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZNEGATIONMT_PROP", new GazProbability("CP", "negation")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZTEXTNOUNMT_PROP", new GazProbability("BOTH", "text_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZTEXTNOUNMT_PROP", new GazProbability("RP", "text_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZTEXTNOUNMT_PROP", new GazProbability("CP", "text_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZPROBLEMADJMT_PROP", new GazProbability("BOTH", "problem_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZPROBLEMADJMT_PROP", new GazProbability("RP", "problem_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZPROBLEMADJMT_PROP", new GazProbability("CP", "problem_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZTHIRDPRONMT_PROP", new GazProbability("BOTH", "third_pron")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZTHIRDPRONMT_PROP", new GazProbability("RP", "third_pron")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZTHIRDPRONMT_PROP", new GazProbability("CP", "third_pron")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZTRADITIONADJMT_PROP", new GazProbability("BOTH", "tradition_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZTRADITIONADJMT_PROP", new GazProbability("RP", "tradition_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZTRADITIONADJMT_PROP", new GazProbability("CP", "tradition_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZPRESENTATIONNOUNMT_PROP", new GazProbability("BOTH", "presentation_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZPRESENTATIONNOUNMT_PROP", new GazProbability("RP", "presentation_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZPRESENTATIONNOUNMT_PROP", new GazProbability("CP", "presentation_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZRESEARCHNOUNMT_PROP", new GazProbability("BOTH", "research_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZRESEARCHNOUNMT_PROP", new GazProbability("RP", "research_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZRESEARCHNOUNMT_PROP", new GazProbability("CP", "research_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZMAINADJMT_PROP", new GazProbability("BOTH", "main_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZMAINADJMT_PROP", new GazProbability("RP", "main_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZMAINADJMT_PROP", new GazProbability("CP", "main_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZREFLEXSIVEMT_PROP", new GazProbability("BOTH", "reflexive")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZREFLEXSIVEMT_PROP", new GazProbability("RP", "reflexive")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZREFLEXSIVEMT_PROP", new GazProbability("CP", "reflexive")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZNEDADJMT_PROP", new GazProbability("BOTH", "ned_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZNEDADJMT_PROP", new GazProbability("RP", "ned_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZNEDADJMT_PROP", new GazProbability("CP", "ned_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZMANYMT_PROP", new GazProbability("BOTH", "many")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZMANYMT_PROP", new GazProbability("RP", "many")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZMANYMT_PROP", new GazProbability("CP", "many")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZCOMPARISONNOUNMT_PROP", new GazProbability("BOTH", "comparison_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZCOMPARISONNOUNMT_PROP", new GazProbability("RP", "comparison_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZCOMPARISONNOUNMT_PROP", new GazProbability("CP", "comparison_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZGOODADJMT_PROP", new GazProbability("BOTH", "good_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZGOODADJMT_PROP", new GazProbability("RP", "good_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZGOODADJMT_PROP", new GazProbability("CP", "good_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZCHANGENOUNMT_PROP", new GazProbability("BOTH", "change_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZCHANGENOUNMT_PROP", new GazProbability("RP", "change_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZCHANGENOUNMT_PROP", new GazProbability("CP", "change_noun")));

            featSet.addFeature(new StringW<TrainingExample, DocumentCtx>("SENTENCEBIGRAMLEMMAS_STRING", new SentenceNGramsStrings("BOTH", "LemmasNGrams", "2-gram")));
            featSet.addFeature(new StringW<TrainingExample, DocumentCtx>("SENTENCELEMMAS_STRING", new SentenceNGramsStrings("BOTH", "LemmasNGrams", "1-gram")));
            featSet.addFeature(new StringW<TrainingExample, DocumentCtx>("SENTENCEBIGRAMPOSS_STRING", new SentenceNGramsStrings("BOTH", "POSNGrams", "2-gram")));
            featSet.addFeature(new StringW<TrainingExample, DocumentCtx>("SENTENCEPOSS_STRING", new SentenceNGramsStrings("BOTH", "POSNGrams", "1-gram")));

            Set<String> facetClassValues = new HashSet<String>();
            if(type.equals("ALL")) {
                facetClassValues.add("Aim_Citation");
                facetClassValues.add("Hypothesis_Citation");
                facetClassValues.add("Method_Citation");
                facetClassValues.add("Results_Citation");
                facetClassValues.add("Implication_Citation");
            }else if(type.equals("METHOD"))
            {
                facetClassValues.add("Others_Citation");
                facetClassValues.add("Method_Citation");

            }else if(type.equals("OTHERS"))
            {
                facetClassValues.add("Aim_Citation");
                facetClassValues.add("Hypothesis_Citation");
                facetClassValues.add("Results_Citation");
                facetClassValues.add("Implication_Citation");
            }

            // Class feature (lasts)
            featSet.addFeature(new NominalW<TrainingExample, DocumentCtx>("class", facetClassValues, new ClassGetter(false)));
        } catch (FeatureException e) {
            System.out.println("Error instantiating feature generation template.");
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return featSet;
    }

    public static void FeatureSetToARFF(FeatureSet<TrainingExample, DocumentCtx> featureSet, String outputPath, String outputInstancesType, String version) {
        try {
            ArffSaver saver = new ArffSaver();
            saver.setInstances(FeatUtil.wekaInstanceGeneration(featureSet, outputInstancesType + " scisumm2017_v_" + version));
            saver.setFile(new File(outputPath + File.separator + "scisumm2017_" + outputInstancesType + "_v_" + version + ".arff"));
            saver.writeBatch();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (FeatSetConsistencyException e) {
            e.printStackTrace();
        }
    }

    public static HashMap<String, Set<String>> GenerateOffsetsMap(String workingDirectory) {

        String offsetsFilePath = workingDirectory + File.separator + "clscisumm2017_offsets_with_negative_samples_w1.csv";
        BufferedReader reader;
        String line;

        HashMap<String, Set<String>> map = new HashMap<String, Set<String>>();
        Set<String> referenceOffsets = new HashSet<String>();
        try {
            reader = new BufferedReader(
                    new InputStreamReader(
                            new FileInputStream(offsetsFilePath), "UTF-8"));

            reader.readLine();
            while ((line = reader.readLine()) != null) {
                if (!line.equals("")) {
                    String[] values = line.split("\t");
                    if (map.containsKey(values[0].trim())) {
                        Set<String> set = map.get(values[0].trim());
                        if (values[3].trim().equals("Not_Referred")) {
                            if (values[2].contains("-")) {
                                for (String offset : values[2].split("-")) {
                                    set.add(offset);
                                }
                            } else {
                                set.add(values[2].trim());
                            }
                        }
                        map.put(values[0].trim(), set);
                    } else {
                        Set<String> set = new HashSet<String>();
                        if (values[3].trim().equals("Not_Referred")) {
                            if (values[2].contains("-")) {
                                for (String offset : values[2].split("-")) {
                                    set.add(offset);
                                }
                            } else {
                                set.add(values[2].trim());
                            }
                        }
                        map.put(values[0].trim(), set);
                    }
                }
            }
        } catch (UnsupportedEncodingException e) {
            e.printStackTrace();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }

        return map;
    }

    public static InputMappedClassifier loadInputMappedClassifier(File classifierModel, File
            classifierDataStructure) {
        // Load classifier
        SerializedClassifier coreClassifier = new SerializedClassifier();
        coreClassifier.setModelFile(classifierModel);
        coreClassifier.setDebug(false);

        // Load InputMappedClassifier and set the just loaded model as classifier
        InputMappedClassifier inputMappedClassifier = new InputMappedClassifier();

        inputMappedClassifier.setClassifier(coreClassifier);

        ConverterUtils.DataSource source = null;

        try {
            source = new ConverterUtils.DataSource(classifierDataStructure.getAbsolutePath());

            Instances headerModel = source.getDataSet();

            headerModel.setClassIndex(headerModel.numAttributes() - 1);
            inputMappedClassifier.setModelHeader(headerModel);
            inputMappedClassifier.setModelPath(classifierModel.getAbsolutePath());

            inputMappedClassifier.setDebug(false);
            inputMappedClassifier.setSuppressMappingReport(true);
            inputMappedClassifier.setTrim(true);
            inputMappedClassifier.setIgnoreCaseForNames(false);

        } catch (Exception e) {
            e.printStackTrace();
        }

        return inputMappedClassifier;
    }

    public static Instances readDataStructure(File dataStructure) {
        BufferedReader reader = null;
        Instances data = null;
        try {
            reader = new BufferedReader(
                    new FileReader(dataStructure));
            data = new Instances(reader);
            reader.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        // setting class attribute
        data.setClassIndex(data.numAttributes() - 1);
        return data;
    }

    public static Instances applyStringToWordVectorFilter(Instances dataSet, String options) {
        Instances instances;
        Instances filteredInstances = null;
        try {
            instances = dataSet;
            instances.setClassIndex(instances.numAttributes() - 1);

            StringToWordVector stringToWordVectorFilter = new StringToWordVector();
            stringToWordVectorFilter.setInputFormat(dataSet);
            stringToWordVectorFilter.setOptions(weka.core.Utils.splitOptions(options));
            filteredInstances = Filter.useFilter(instances, stringToWordVectorFilter);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return filteredInstances;
    }

    public static Instances applyReorderFilter(Instances dataSet, String options) {
        Instances instances;
        Instances filteredInstances = null;
        try {
            instances = dataSet;
            instances.setClassIndex(instances.numAttributes() - 1);

            Reorder reorderFilter = new Reorder();
            reorderFilter.setOptions(weka.core.Utils.splitOptions(options));
            reorderFilter.setInputFormat(dataSet);
            filteredInstances = Filter.useFilter(instances, reorderFilter);
            filteredInstances.setClassIndex(filteredInstances.numAttributes() - 1);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return filteredInstances;
    }

    public static Instances classifyInstances(Instances testData, InputMappedClassifier
            inputMappedClassifier) {
        try {
            //System.out.println("Total number of testing instances : " + testData.size());
            for (int i = 0; i < testData.numInstances(); i++) {
                double pred = inputMappedClassifier.classifyInstance(testData.instance(i));
                testData.instance(i).setClassValue(pred);
            }

        } catch (Exception e) {
            e.printStackTrace();
        }

        testData.setClassIndex(testData.numAttributes() - 1);
        return testData;
    }

    public static void printSciSummOutput(HashMap<String, SciSummAnnotation> output, String filePath) {
        for (String key : output.keySet()) {
            SciSummAnnotation sciSummAnnotation = output.get(key);
            StringBuilder stringBuilder = new StringBuilder();

            stringBuilder.append("Citance Number: ");
            stringBuilder.append(sciSummAnnotation.getCitance_Number());
            stringBuilder.append(" | Reference Article: ");
            stringBuilder.append(sciSummAnnotation.getReference_Article());
            stringBuilder.append(" | Citing Article: ");
            stringBuilder.append(sciSummAnnotation.getCiting_Article());
            stringBuilder.append(" | Citation Marker Offset:  ['");
            stringBuilder.append(sciSummAnnotation.getCitation_Marker_Offset());
            stringBuilder.append("'] | Citation Marker: ");
            stringBuilder.append(sciSummAnnotation.getCitation_Marker());
            stringBuilder.append(" | Citation Offset:  [");

            for(int i=0; i < sciSummAnnotation.getCitation_Offset().size();i++)
            {
                stringBuilder.append("'" + sciSummAnnotation.getCitation_Offset().get(i) + "'");
                if(i!= sciSummAnnotation.getCitation_Offset().size() -1)
                {
                    stringBuilder.append(",");
                }
            }
            stringBuilder.append("] | Citation Text: ");
            stringBuilder.append(sciSummAnnotation.getCitation_Text());
            stringBuilder.append(" | Reference Offset:  [");
            for(int i=0; i< sciSummAnnotation.getReference_Offset().size();i++)
            {
                stringBuilder.append("'" + sciSummAnnotation.getReference_Offset().get(i) + "'");
                if(i!= sciSummAnnotation.getReference_Offset().size() -1)
                {
                    stringBuilder.append(",");
                }
            }
            stringBuilder.append("] | Reference Text: ");
            stringBuilder.append(sciSummAnnotation.getReference_Text());
            stringBuilder.append(" | Discourse Facet:  [");
            for(int i=0; i< sciSummAnnotation.getDiscourse_Facet().size();i++)
            {
                stringBuilder.append("'" + sciSummAnnotation.getDiscourse_Facet().get(i) + "'");
                if(i!= sciSummAnnotation.getDiscourse_Facet().size() -1)
                {
                    stringBuilder.append(",");
                }
            }
            stringBuilder.append("] | Annotator: ");
            stringBuilder.append(sciSummAnnotation.getAnnotator());
            stringBuilder.append(" |");
            stringBuilder.append(System.getProperty("line.separator"));

            File file = new File(filePath);
            if(file.exists())
            {
                try (FileWriter fw = new FileWriter(file, true);
                     BufferedWriter bw = new BufferedWriter(fw);
                     PrintWriter out = new PrintWriter(bw)) {
                    out.println(stringBuilder.toString());
                    //more code

                } catch (IOException e) {
                    //exception handling left as an exercise for the reader
                }
            }else {
                BufferedWriter writer = null;
                try {
                    writer = new BufferedWriter(new FileWriter(file));

                    writer.write(stringBuilder.toString());
                    writer.newLine();
                    writer.flush();
                    writer.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    public static HashMap<String, SciSummAnnotation> fillOffsetsSenttences(HashMap<String, SciSummAnnotation> output, HashMap<String, Document> documents) throws InvalidOffsetException {
        for(String key: output.keySet())
        {
            SciSummAnnotation sciSummAnnotation = output.get(key);
            Document referenceDoc = documents.get(sciSummAnnotation.getReference_Article());
            Document citingDoc = documents.get(sciSummAnnotation.getCiting_Article());

            AnnotationSet referenceSentences = referenceDoc.getAnnotations("Original markups").get("S");
            AnnotationSet citingSentences = citingDoc.getAnnotations("Original markups").get("S");

            StringBuilder referenceText = new StringBuilder();
            StringBuilder citingText = new StringBuilder();

            Iterator referenceSentencesIterator = referenceSentences.iterator();
            while(referenceSentencesIterator.hasNext())
            {
                Annotation referenceSentence = (Annotation) referenceSentencesIterator.next();
                if(sciSummAnnotation.getReference_Offset().contains(referenceSentence.getFeatures().get("sid").toString()))
                {
                    referenceText.append("<S sid =\"" + referenceSentence.getFeatures().get("sid").toString() + "\" ssid = \"" + referenceSentence.getFeatures().get("ssid").toString() + "\">" + referenceDoc.getContent().getContent(referenceSentence.getStartNode().getOffset(), referenceSentence.getEndNode().getOffset()).toString().replaceAll("\\|", "") + "</S>");
                }
            }
            Iterator citingSentencesIterator = citingSentences.iterator();
            while(citingSentencesIterator.hasNext())
            {
                Annotation citingSentence = (Annotation) citingSentencesIterator.next();
                if(sciSummAnnotation.getCitation_Offset().contains(citingSentence.getFeatures().get("sid").toString()))
                {
                    citingText.append("<S sid =\"" + citingSentence.getFeatures().get("sid").toString() + "\" ssid = \"" + citingSentence.getFeatures().get("ssid").toString() + "\">" + citingDoc.getContent().getContent(citingSentence.getStartNode().getOffset(), citingSentence.getEndNode().getOffset()).toString().replaceAll("\\|", "") + "</S>");
                }
            }
            sciSummAnnotation.setReference_Text(referenceText.toString());
            sciSummAnnotation.setCitation_Text(citingText.toString());
            output.put(key, sciSummAnnotation);
        }

        return output;
    }
}
