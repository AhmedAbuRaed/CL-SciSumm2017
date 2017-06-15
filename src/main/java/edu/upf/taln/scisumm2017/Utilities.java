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

import com.fasterxml.jackson.core.JsonParseException;
import com.fasterxml.jackson.databind.JsonMappingException;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.*;
import java.net.MalformedURLException;
import java.net.URL;
import java.net.URLEncoder;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;

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
                        if (fields.length == 8) {
                            if (fields[7].trim().split(":")[1].trim().contains(",")) {
                                annotation.setAnnotator(fields[7].trim().split(":")[1].trim()
                                        .substring(0, fields[7].trim().split(":")[1].trim().indexOf(",")).replaceAll(" ", "_"));
                            } else {
                                annotation.setAnnotator(fields[7].trim().split(":")[1].trim().replaceAll(" ", "_"));
                            }
                        } else {
                            if (fields[10].trim().split(":")[1].trim().contains(",")) {
                                annotation.setAnnotator(fields[10].trim().split(":")[1].trim()
                                        .substring(0, fields[10].trim().split(":")[1].trim().indexOf(",")).replaceAll(" ", "_"));
                            } else {
                                annotation.setAnnotator(fields[10].trim().split(":")[1].trim().replaceAll(" ", "_"));
                            }
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
            String annotator = annotation.getAnnotator();

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


}
