package edu.upf.taln.scisumm2017.feature.calculator;

import edu.upf.taln.ml.feat.base.FeatCalculator;
import edu.upf.taln.ml.feat.base.MyDouble;
import edu.upf.taln.scisumm2017.feature.context.DocumentCtx;
import edu.upf.taln.scisumm2017.reader.TrainingExample;
import gate.Annotation;
import gate.AnnotationSet;
import gate.Document;

/**
 * Created by ahmed on 5/17/2016.
 */
public class CosineSimilarity implements FeatCalculator<Double, TrainingExample, DocumentCtx> {
    String cosineType;

    public CosineSimilarity(String cosineType) {
        this.cosineType = cosineType;
    }

    @Override
    public MyDouble calculateFeature(TrainingExample obj, DocumentCtx docs, String CosineSimilarity) {
        MyDouble value = new MyDouble(0d);

        try {
            Document rp = docs.getReferenceDoc();
            Document cp = docs.getCitationDoc();

            Annotation refSentence = obj.getReferenceSentence();
            Annotation citSentence = obj.getCitanceSentence();

            AnnotationSet rpSimilarities = rp.getAnnotations("Analysis").get("Sentence").get(refSentence.getStartNode().getOffset(),
                    refSentence.getEndNode().getOffset());

            AnnotationSet rpBabelnetSimilarities = rp.getAnnotations("Babelnet").get("Sentence").get(refSentence.getStartNode().getOffset(),
                    refSentence.getEndNode().getOffset());

            AnnotationSet cpAnnotators = cp.getAnnotations("CITATIONS").get(citSentence.getStartNode().getOffset(),
                    citSentence.getEndNode().getOffset());

            if (cosineType.equals("LEMMA")) {
                if (rpSimilarities.size() > 0 && cpAnnotators.size() > 0) {
                    Annotation rpSentence = rpSimilarities.iterator().next();
                    Annotation cpAnnotator = cpAnnotators.iterator().next();
                    value.setValue((Double) rpSentence.getFeatures().get("sim_" + cpAnnotator.getFeatures().get("id")));
                }
            } else if (cosineType.equals("BABELNET")) {
                if (rpBabelnetSimilarities.size() > 0 && cpAnnotators.size() > 0) {
                    Annotation rpSentence = rpBabelnetSimilarities.iterator().next();
                    Annotation cpAnnotator = cpAnnotators.iterator().next();
                    value.setValue((Double) rpSentence.getFeatures().get("BNsim_" + cpAnnotator.getFeatures().get("id")));
                }
            }


        } catch (Exception e) {
            e.printStackTrace();
        }

        return value;
    }
}
