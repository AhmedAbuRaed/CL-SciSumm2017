package edu.upf.taln.scisumm2017.feature.calculator;

import edu.upf.taln.ml.feat.base.FeatCalculator;
import edu.upf.taln.scisumm2017.feature.context.DocumentCtx;
import edu.upf.taln.scisumm2017.reader.TrainingExample;
import gate.Document;
import edu.upf.taln.ml.feat.base.MyDouble;
import gate.Annotation;
import gate.AnnotationSet;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

/**
 *
 * @author Pablo
 */
public class Jaccard implements FeatCalculator<Double, TrainingExample, DocumentCtx> {
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
     * @param trimLengthLemma
     * @param minLengthLemma
     * @param significativePos
     * @param stopWords 
     */
    public Jaccard(int trimLengthLemma, int minLengthLemma, String[] significativePos, String[] stopWords) {
        super();        
        this.trimLengthLemma = trimLengthLemma;
        this.minLengthLemma = minLengthLemma;
        this.significativePos = significativePos;
        this.stopWords = stopWords;        
    }
    
    /**
     * 
     * @param trimLengthLemma
     * @param minLengthLemma 
     */
    public Jaccard(int trimLengthLemma, int minLengthLemma) {
        super();        
        this.trimLengthLemma = trimLengthLemma;
        this.minLengthLemma = minLengthLemma;
        this.significativePos = new String[]{"JJ", "JJR", "JJS", "NN", "NNS", "NNP", "NNPS", "RB", "RBR", "RBS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"};            
        this.stopWords = new String[]{"have", "be", "paper", "present", "describe"};         
    }   
    
    /**
     * 
     */
    public Jaccard() {
        super();        
        this.significativePos = new String[]{"JJ", "JJR", "JJS", "NN", "NNS", "NNP", "NNPS", "RB", "RBR", "RBS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"};            
        this.stopWords = new String[]{"have", "be", "paper", "present", "describe"}; 
        this.trimLengthLemma = 8;
        this.minLengthLemma = 2;  
    }    
    
    @Override    
    public MyDouble calculateFeature(TrainingExample trainingExample, DocumentCtx documentCtx, String string) {
        MyDouble retValue = new MyDouble(0d);
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
                    citLemma = citLemma.substring(0, Math.min(trimLengthLemma, citLemma.length()));
                    citDocLemmas.add(citLemma);  
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
                    refLemma = refLemma.substring(0, Math.min(trimLengthLemma, refLemma.length()));                
                    refDocLemmas.add(refLemma);            
                }
            }        
        }      
        // Calculate the Jaccard index
        if (!citDocLemmas.isEmpty() && !refDocLemmas.isEmpty()) {            
            // Get intersection
            Set<String> intersection = new HashSet<>(citDocLemmas);
            intersection.retainAll(refDocLemmas);
            // Get union
            Set<String> union = new HashSet<>(citDocLemmas);
            union.addAll(refDocLemmas); 
            double jaccard = (double) intersection.size() / (double) union.size();
            retValue.setValue(jaccard);                            
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
