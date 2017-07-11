package edu.upf.taln.scisumm2017.feature.calculator;

import edu.upf.taln.ml.feat.base.FeatCalculator;
import edu.upf.taln.ml.feat.base.MyDouble;
import edu.upf.taln.scisumm2017.feature.context.DocumentCtx;
import edu.upf.taln.scisumm2017.reader.TrainingExample;
import gate.Annotation;
import gate.AnnotationSet;
import gate.Document;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 *
 * @author Pablo
 */
public class IdfWeightedJaccard implements FeatCalculator<Double, TrainingExample, DocumentCtx> {
    // Significative POS to count. Leave empty to consider all.
    private final String[] significativePos;
    // Stop-words to filter out.
    private final String[] stopWords;
    // Length of the maximum prefix considered for each lemma.
    private final int trimLengthLemma;
    // Minimum length of the lemmas counted.
    private final int minLengthLemma;
    
    /**
     * 
     * @param IdfTable
     * @param trimLengthLemma
     * @param minLengthLemma
     * @param significativePos
     * @param stopWords 
     */
    public IdfWeightedJaccard(int trimLengthLemma, int minLengthLemma, String[] significativePos, String[] stopWords) {
        super();   
        this.trimLengthLemma = trimLengthLemma;
        this.minLengthLemma = minLengthLemma;
        this.significativePos = significativePos;
        this.stopWords = stopWords;        
    }
    
    /**
     * 
     * @param IdfTable
     * @param trimLengthLemma
     * @param minLengthLemma 
     */
    public IdfWeightedJaccard(int trimLengthLemma, int minLengthLemma) {
        super();       
        this.trimLengthLemma = trimLengthLemma;
        this.minLengthLemma = minLengthLemma;
        this.significativePos = new String[]{"JJ", "JJR", "JJS", "NN", "NNS", "NNP", "NNPS", "RB", "RBR", "RBS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"};            
        this.stopWords = new String[]{"have", "be", "paper", "present", "describe"};         
    }   
    
    /**
     * 
     * @param IdfTable
     */
    public IdfWeightedJaccard() {
        super();        
        this.significativePos = new String[]{"JJ", "JJR", "JJS", "NN", "NNS", "NNP", "NNPS", "RB", "RBR", "RBS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"};            
        this.stopWords = new String[]{"have", "be", "paper", "present", "describe"}; 
        this.trimLengthLemma = 8;
        this.minLengthLemma = 2;  
    }     
    
    @Override
    public MyDouble calculateFeature(TrainingExample trainingExample, DocumentCtx documentCtx, String string) {
        MyDouble retValue = new MyDouble(null);
        Map<String, Double> idfValuesLemmaPrefixes = new HashMap<>();
        List<String> significativePosList = Arrays.asList(significativePos);
        List<String> stopWordsList = Arrays.asList(stopWords);           
        // Get GATE documents
        Document citDoc = documentCtx.getCitationDoc();
        Document refDoc = documentCtx.getReferenceDoc();
        // Get citation and reference sentences
        Annotation citSentence = trainingExample.getCitanceSentence();
        Annotation refSentence = trainingExample.getReferenceSentence();
        // Get tokens within sentences
        AnnotationSet citDocTokens = citDoc.getAnnotations("Analysis").get("Token").getContained(citSentence.getStartNode().getOffset(), citSentence.getEndNode().getOffset());        
        AnnotationSet refDocTokens = refDoc.getAnnotations("Analysis").get("Token").getContained(refSentence.getStartNode().getOffset(), refSentence.getEndNode().getOffset());
        // Get citation spans in citation sentence
        AnnotationSet citSentenceCitationSpans = citDoc.getAnnotations("Analysis").get("CitSpan").get(citSentence.getStartNode().getOffset(), citSentence.getEndNode().getOffset());
        AnnotationSet refSentenceCitationSpans = refDoc.getAnnotations("Analysis").get("CitSpan").get(refSentence.getStartNode().getOffset(), refSentence.getEndNode().getOffset());
        // Get citation sentence significative lemmas not contained in citation spans.
        Set<String> citDocLemmas = new HashSet<>();     
        for (Annotation citDocToken : citDocTokens) {
            String category = (String) citDocToken.getFeatures().get("category");
            if ((significativePosList.isEmpty() || significativePosList.contains(category)) && !isContainedInCitations(citDocToken, citSentenceCitationSpans)) {
                String citLemma = (String) citDocToken.getFeatures().get("lemma");
                if (!stopWordsList.contains(citLemma) && citLemma.length() >= minLengthLemma) {
                    Double idfLemma = Double.valueOf((String) citDocToken.getFeatures().get("token_idf"));                    
                    citLemma = citLemma.substring(0, Math.min(trimLengthLemma, citLemma.length()));
                    citDocLemmas.add(citLemma); 
                    Double idfLemmaPrefix = idfValuesLemmaPrefixes.get(citLemma);
                    if (idfLemmaPrefix != null) {
                        idfLemmaPrefix = (idfLemma + idfLemmaPrefix) / (double) 2;
                    }
                    else {
                        idfLemmaPrefix = idfLemma;
                    }
                    idfValuesLemmaPrefixes.put(citLemma, idfLemmaPrefix);
                }
            }
        }        
        // Get reference sentence significative lemmas not contained in citation spans.
        Set<String> refDocLemmas = new HashSet<>();
        for (Annotation refDocToken : refDocTokens) {
            String category = (String) refDocToken.getFeatures().get("category");
            if ((significativePosList.isEmpty() || significativePosList.contains(category)) && !isContainedInCitations(refDocToken, refSentenceCitationSpans)) {
                String refLemma = (String) refDocToken.getFeatures().get("lemma");
                if (!stopWordsList.contains(refLemma) && refLemma.length() >= minLengthLemma) {
                    Double idfLemma = Double.valueOf((String) refDocToken.getFeatures().get("token_idf"));                                        
                    refLemma = refLemma.substring(0, Math.min(trimLengthLemma, refLemma.length()));
                    refDocLemmas.add(refLemma);
                    Double idfLemmaPrefix = idfValuesLemmaPrefixes.get(refLemma);
                    if (idfLemmaPrefix != null) {
                        idfLemmaPrefix = (idfLemma + idfLemmaPrefix) / (double) 2;
                    } else {
                        idfLemmaPrefix = idfLemma;
                    }
                    idfValuesLemmaPrefixes.put(refLemma, idfLemmaPrefix);
                }
            }
        }    
        // Calculate the Jaccard index
        if (!citDocLemmas.isEmpty() && !refDocLemmas.isEmpty()) {            
            // Get weighted intersection value
            double weightedIntersection = 0;
            for (String lemmaPrefix : citDocLemmas) {
                if (refDocLemmas.contains(lemmaPrefix)) {
                    Double idfLemmaPrefix = idfValuesLemmaPrefixes.get(lemmaPrefix);
                    double powerIdfLemmaPrefix = Math.pow(idfLemmaPrefix, 2);
                    weightedIntersection += powerIdfLemmaPrefix;
                }
            }
            // Get union            
            Set<String> union = new HashSet<>(citDocLemmas);
            union.addAll(refDocLemmas); 
            double weightedJaccard = weightedIntersection / (double) union.size();
            retValue.setValue(weightedJaccard);                            
        }    
        return retValue;
    }
    
    /**
     * 
     * @param token
     * @param citationSpans
     * @return 
     */
    private boolean isContainedInCitations(Annotation token, AnnotationSet citationSpans) {
        boolean isContained = false;
        Iterator<Annotation> iterCitations = citationSpans.iterator();
        while (!isContained && iterCitations.hasNext()) {
            isContained = token.withinSpanOf((Annotation) iterCitations.next());
        }
        return isContained;
    }
    
}
