package edu.upf.taln.scisumm2017.feature.calculator;

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
public class CoRefChainsCount implements FeatCalculator<Double, TrainingExample, DocumentCtx> {
    boolean normalized;
    String target;

    public CoRefChainsCount(boolean normalized, String target) {
        this.normalized = normalized;
        this.target = target;
    }

    @Override
    public MyDouble calculateFeature(TrainingExample obj, DocumentCtx docs, String CoRefChainsCount) {
        MyDouble value = new MyDouble(0d);
        double val = 0;
        double totalCount = 0;

        try
        {
            Document rp = docs.getReferenceDoc();
            Document cp = docs.getCitationDoc();

            Annotation refSentence = obj.getReferenceSentence();
            Annotation citSentence = obj.getCitanceSentence();

            double rpTotalCount = rp.getAnnotations("CorefChains").size();
            double cpTotalCount = cp.getAnnotations("CorefChains").size();

            AnnotationSet rpCorefChains = rp.getAnnotations("CorefChains").get(refSentence.getStartNode().getOffset(),
                    refSentence.getEndNode().getOffset());

            AnnotationSet cpCorefChains = cp.getAnnotations("CorefChains").get(citSentence.getStartNode().getOffset(),
                    citSentence.getEndNode().getOffset());

            switch (target) {
                case "RP":
                    for (Annotation annotation : rpCorefChains) {
                        val++;
                    }
                    totalCount = rpTotalCount;
                    break;
                case "CP":
                    for (Annotation annotation : cpCorefChains) {
                        val++;
                    }
                    totalCount = cpTotalCount;
                    break;
                case "BOTH":
                    for (Annotation annotation : rpCorefChains) {
                        val++;
                    }

                    for (Annotation annotation : cpCorefChains) {
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
        }
        catch (Exception e)
        {
            e.printStackTrace();
        }

        return value;
    }
}
