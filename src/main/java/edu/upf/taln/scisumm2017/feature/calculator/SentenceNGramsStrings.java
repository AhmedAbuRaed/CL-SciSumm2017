package edu.upf.taln.scisumm2017.feature.calculator;

import edu.upf.taln.ml.feat.base.FeatCalculator;
import edu.upf.taln.ml.feat.base.MyString;
import edu.upf.taln.scisumm2017.feature.context.DocumentCtx;
import edu.upf.taln.scisumm2017.reader.TrainingExample;
import gate.Annotation;
import gate.Document;

/**
 * Created by ahmed on 7/8/16.
 */
public class SentenceNGramsStrings implements FeatCalculator<String, TrainingExample, DocumentCtx> {
    String target;
    String nGramsASName;
    String nGramsTypeASName;

    public SentenceNGramsStrings(String target, String nGramsASName, String nGramsTypeASName)
    {
        this.target = target;
        this.nGramsASName = nGramsASName;
        this.nGramsTypeASName = nGramsTypeASName;
    }

    @Override
    public MyString calculateFeature(TrainingExample obj, DocumentCtx docs, String SentenceLemmasString) {
        MyString value = new MyString("");
        String val="";

        Document rp = null;
        Document cp = null;

        try
        {
            rp = docs.getReferenceDoc();
            cp = docs.getCitationDoc();

            Annotation refSentence = obj.getReferenceSentence();
            Annotation citSentence = obj.getCitanceSentence();

            switch (target) {
                case "RP":
                    for (Annotation annotation : rp.getAnnotations(nGramsTypeASName).get(nGramsASName).get(refSentence.getStartNode().getOffset(), refSentence.getEndNode().getOffset())) {
                        val = val + " " + annotation.getFeatures().get("string").toString().replaceAll(" ", "_");
                    }
                    break;
                case "CP":
                    for (Annotation annotation : cp.getAnnotations(nGramsTypeASName).get(nGramsASName).get(citSentence.getStartNode().getOffset(), citSentence.getEndNode().getOffset())) {
                        val = val + " " + annotation.getFeatures().get("string").toString().replaceAll(" ", "_");
                    }
                    break;
                case "BOTH":
                    for (Annotation annotation : rp.getAnnotations(nGramsTypeASName).get(nGramsASName).get(refSentence.getStartNode().getOffset(), refSentence.getEndNode().getOffset())) {
                        val = val + " " + annotation.getFeatures().get("string").toString().replaceAll(" ", "_");
                    }
                    for (Annotation annotation : cp.getAnnotations(nGramsTypeASName).get(nGramsASName).get(citSentence.getStartNode().getOffset(), citSentence.getEndNode().getOffset())) {
                        val = val + " " + annotation.getFeatures().get("string").toString().replaceAll(" ", "_");
                    }
                    break;
            }
            value.setValue(val);
        }
        catch (Exception e)
        {
            e.printStackTrace();
        }
        return value;
    }
}
