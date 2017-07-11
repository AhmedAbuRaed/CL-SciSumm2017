package edu.upf.taln.scisumm2017;

import edu.upf.taln.scisumm2017.reader.BabelnetSynset;
import edu.upf.taln.scisumm2017.reader.SciSummAnnotation;
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

}
