package edu.upf.taln.scisumm2017.feature.calculator;

import edu.cmu.lti.lexical_db.ILexicalDatabase;
import edu.cmu.lti.lexical_db.NictWordNet;
import edu.upf.taln.ml.feat.base.FeatCalculator;
import edu.upf.taln.ml.feat.base.MyDouble;
import edu.upf.taln.scisumm2017.feature.context.DocumentCtx;
import edu.upf.taln.scisumm2017.reader.TrainingExample;
import gate.Annotation;
import gate.AnnotationSet;
import gate.Document;

/**
 * Created by ahmed on 7/8/16.
 */
public class CitationMarkerCount implements FeatCalculator<Double, TrainingExample, DocumentCtx> {
    boolean normalized;
    String target;

    public CitationMarkerCount(boolean normalized, String target) {
        this.normalized = normalized;
        this.target = target;
    }

    @Override
    public MyDouble calculateFeature(TrainingExample obj, DocumentCtx docs, String CitationMarkerCount) {
        MyDouble value = new MyDouble(0d);
        double val = 0;
        double totalCount = 0;

        try {
            Document rp = docs.getReferenceDoc();
            Document cp = docs.getCitationDoc();

            Annotation refSentence = obj.getReferenceSentence();
            Annotation citSentence = obj.getCitanceSentence();

            double rpTotalCount = rp.getAnnotations("Analysis").get("CitMarker").size();
            double cpTotalCount = cp.getAnnotations("Analysis").get("CitMarker").size();

            AnnotationSet rpCitMarkers = rp.getAnnotations("Analysis").get("CitMarker").get(refSentence.getStartNode().getOffset(),
                    refSentence.getEndNode().getOffset());

            AnnotationSet cpCitMarkers = cp.getAnnotations("Analysis").get("CitMarker").get(citSentence.getStartNode().getOffset(),
                    citSentence.getEndNode().getOffset());

            switch (target) {
                case "RP":
                    for (Annotation annotation : rpCitMarkers) {
                        val++;
                    }
                    totalCount = rpTotalCount;
                    break;
                case "CP":
                    for (Annotation annotation : cpCitMarkers) {
                        val++;
                    }
                    totalCount = cpTotalCount;
                    break;
                case "BOTH":
                    for (Annotation annotation : rpCitMarkers) {
                        val++;
                    }

                    for (Annotation annotation : cpCitMarkers) {
                        val++;
                    }
                    totalCount = rpTotalCount + cpTotalCount;
                    break;
            }

            if (normalized) {
                if (totalCount > 0) {
                    value.setValue(val / totalCount);
                }
            } else {
                value.setValue(val);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

        return value;
    }
}
