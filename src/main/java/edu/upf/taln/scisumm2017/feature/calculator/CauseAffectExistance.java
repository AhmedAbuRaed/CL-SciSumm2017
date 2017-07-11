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
public class CauseAffectExistance implements FeatCalculator<Double, TrainingExample, DocumentCtx> {
    String target;

    public CauseAffectExistance(boolean normalized, String target) {
        this.target = target;
    }

    @Override
    public MyDouble calculateFeature(TrainingExample obj, DocumentCtx docs, String CauseAffectExistance) {
        MyDouble value = new MyDouble(0d);

        try
        {
            switch (target) {
                case "RP":
                    Document rp = docs.getReferenceDoc();

                    Annotation refSentence = obj.getReferenceSentence();

                    AnnotationSet rpCause = rp.getAnnotations("Causality").get("CAUSE").get(refSentence.getStartNode().getOffset(),
                            refSentence.getEndNode().getOffset());

                    AnnotationSet rpEffect = rp.getAnnotations("Causality").get("EFFECT").get(refSentence.getStartNode().getOffset(),
                            refSentence.getEndNode().getOffset());

                    if ((rpCause.size() > 0 || rpEffect.size() > 0)) {
                        value.setValue(1d);
                    }
                    break;
                case "CP":
                    Document cp = docs.getCitationDoc();

                    Annotation citSentence = obj.getCitanceSentence();

                    AnnotationSet cpCause = cp.getAnnotations("Causality").get("CAUSE").get(citSentence.getStartNode().getOffset(),
                            citSentence.getEndNode().getOffset());

                    AnnotationSet cpEffect = cp.getAnnotations("Causality").get("EFFECT").get(citSentence.getStartNode().getOffset(),
                            citSentence.getEndNode().getOffset());

                    if (cpCause.size() > 0 || cpEffect.size() > 0) {
                        value.setValue(1d);
                    }
                    break;
            }
        }
        catch (Exception e)
        {
            e.printStackTrace();
        }

        return value;
    }
}
