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
 * Created by ahmed on 5/11/2016.
 */
public class SectionTitleFacet implements FeatCalculator<Double, TrainingExample, DocumentCtx> {
    String [] cueWords;

    public SectionTitleFacet(String [] cueWords) throws IOException {
        this.cueWords = cueWords;
    }

    @Override
    public MyDouble calculateFeature(TrainingExample obj, DocumentCtx docs, String SectionTitleFacet) {
        MyDouble value = new MyDouble(0d);

        Document rp = docs.getReferenceDoc();
        Document cp = docs.getCitationDoc();

        Annotation refSentence = obj.getReferenceSentence();
        Annotation citSentence = obj.getCitanceSentence();

        AnnotationSet sections = rp.getAnnotations("Original markups").get("SECTION").get(refSentence.getStartNode().getOffset(),
                refSentence.getEndNode().getOffset());

        if (sections.size() > 0) {
            Annotation section = sections.iterator().next();

            for (String s : this.cueWords) {
                if (section.getFeatures().get("title").toString().toLowerCase().contains(s)) {
                    value.setValue(1d);
                }
            }
        }
        return value;
    }
}
