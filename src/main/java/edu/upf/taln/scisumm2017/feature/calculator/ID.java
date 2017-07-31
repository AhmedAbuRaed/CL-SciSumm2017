package edu.upf.taln.scisumm2017.feature.calculator;

import edu.upf.taln.ml.feat.base.FeatCalculator;
import edu.upf.taln.ml.feat.base.MyString;
import edu.upf.taln.scisumm2017.feature.context.DocumentCtx;
import edu.upf.taln.scisumm2017.reader.TrainingExample;
import gate.Annotation;

import gate.AnnotationSet;
import gate.Document;

/**
 * Created by ahmed on 5/10/2016.
 */
public class ID implements FeatCalculator<String, TrainingExample, DocumentCtx> {
    @Override
    public MyString calculateFeature(TrainingExample obj, DocumentCtx docs, String ID) {
        MyString value = new MyString("");

        Document rp = docs.getReferenceDoc();
        Document cp = docs.getCitationDoc();

        Annotation refSentence = obj.getReferenceSentence();
        Annotation citSentence = obj.getCitanceSentence();

        AnnotationSet cpCitingIDs = cp.getAnnotations("CITATIONS").get(citSentence.getStartNode().getOffset(), citSentence.getEndNode().getOffset());
        if(cpCitingIDs.size()>0)
        {
            Annotation cpID = cpCitingIDs.iterator().next();
            value.setValue(cpID.getFeatures().get("id").toString());
        }

        return value;
    }
}
