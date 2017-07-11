package edu.upf.taln.scisumm2017.feature.calculator;

import edu.upf.taln.ml.feat.base.FeatCalculator;
import edu.upf.taln.ml.feat.base.MyDouble;
import edu.upf.taln.scisumm2017.feature.context.DocumentCtx;
import edu.upf.taln.scisumm2017.reader.TrainingExample;
import gate.Annotation;
import gate.AnnotationSet;
import gate.Document;

import java.io.IOException;

/**
 * Created by ahmed on 5/17/2016.
 */
public class DrInventorFacetProbability implements FeatCalculator<Double, TrainingExample, DocumentCtx> {
    String drInventorFacet;

    public DrInventorFacetProbability(String drInventorFacet) throws IOException {
        this.drInventorFacet = drInventorFacet;
    }

    @Override
    public MyDouble calculateFeature(TrainingExample obj, DocumentCtx docs, String DrInventorFacetProbability) {
        MyDouble value = new MyDouble(0d);

        try
        {
            Document rp = docs.getReferenceDoc();
            Document cp = docs.getCitationDoc();

            Annotation refSentence = obj.getReferenceSentence();
            Annotation citSentence = obj.getCitanceSentence();

            AnnotationSet rpSentenceProbabilities = rp.getAnnotations("Analysis").get("Sentence_LOA").get(refSentence.getStartNode().getOffset(),
                    refSentence.getEndNode().getOffset());

            if (rpSentenceProbabilities.size() > 0) {
                Annotation rpSentence = rpSentenceProbabilities.iterator().next();

                if (rpSentence.getFeatures().get(drInventorFacet) != null)
                    value.setValue((double) rpSentence.getFeatures().get(drInventorFacet));
            }
        }
        catch (Exception e)
        {
            e.printStackTrace();
        }

        return value;
    }
}
