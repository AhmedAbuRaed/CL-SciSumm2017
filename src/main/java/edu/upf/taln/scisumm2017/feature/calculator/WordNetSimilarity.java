package edu.upf.taln.scisumm2017.feature.calculator;

import edu.cmu.lti.lexical_db.ILexicalDatabase;
import edu.cmu.lti.lexical_db.NictWordNet;
import edu.cmu.lti.ws4j.impl.*;
import edu.cmu.lti.ws4j.util.WS4JConfiguration;
import edu.upf.taln.ml.feat.base.FeatCalculator;
import edu.upf.taln.ml.feat.base.MyDouble;
import edu.upf.taln.scisumm2017.feature.context.DocumentCtx;
import edu.upf.taln.scisumm2017.reader.TrainingExample;
import gate.Annotation;
import gate.AnnotationSet;
import gate.Document;

/**
 * Created by ahmed on 5/10/2016.
 */
public class WordNetSimilarity implements FeatCalculator<Double, TrainingExample, DocumentCtx> {
    boolean normalized;
    String similarityMeasure;
    ILexicalDatabase db;

    public WordNetSimilarity(boolean normalized, String similarityMeasure) {
        this.normalized = normalized;
        this.similarityMeasure = similarityMeasure;
        db = new NictWordNet();
    }

    @Override
    public MyDouble calculateFeature(TrainingExample obj, DocumentCtx docs, String WordNetSimilarity) {
        MyDouble value = new MyDouble(0d);

        int count = 0;
        double sum = 0d;

        Document rp = docs.getReferenceDoc();
        Document cp = docs.getCitationDoc();

        Annotation refSentence = obj.getReferenceSentence();
        Annotation citSentence = obj.getCitanceSentence();

        AnnotationSet rpTokens = rp.getAnnotations("Analysis").get("Token").get(refSentence.getStartNode().getOffset(),
                refSentence.getEndNode().getOffset());

        AnnotationSet cpTokens = cp.getAnnotations("Analysis").get("Token").get(citSentence.getStartNode().getOffset(),
                citSentence.getEndNode().getOffset());

        WS4JConfiguration.getInstance().setMFS(true);
        double val;
        for (Annotation cpToken : cpTokens) {
            for (Annotation rpToken : rpTokens) {

                if (!rpToken.getFeatures().get("string").toString().equals(cpToken.getFeatures().get("string").toString())) {
                    switch (similarityMeasure) {
                        case "jiangconrath":
                            if (normalized) {
                                double maxSim = new JiangConrath(db).calcRelatednessOfWords(cpToken.getFeatures().get("string").toString(),
                                        cpToken.getFeatures().get("string").toString());

                                val = new JiangConrath(db).calcRelatednessOfWords(cpToken.getFeatures().get("string").toString(),
                                        rpToken.getFeatures().get("string").toString()) / maxSim;
                            } else {
                                val = new JiangConrath(db).calcRelatednessOfWords(cpToken.getFeatures().get("string").toString(),
                                        rpToken.getFeatures().get("string").toString());
                            }
                            if (!Double.isInfinite(val)) {
                                sum += val;
                                count++;
                            }
                            break;
                        case "lch":
                            if (normalized) {
                                double maxSim = new LeacockChodorow(db).calcRelatednessOfWords(cpToken.getFeatures().get("string").toString(),
                                        cpToken.getFeatures().get("string").toString());

                                val = new LeacockChodorow(db).calcRelatednessOfWords(cpToken.getFeatures().get("string").toString(),
                                        rpToken.getFeatures().get("string").toString()) / maxSim;
                            } else {
                                val = new LeacockChodorow(db).calcRelatednessOfWords(cpToken.getFeatures().get("string").toString(),
                                        rpToken.getFeatures().get("string").toString());
                            }
                            if (!Double.isInfinite(val)) {
                                sum += val;
                                count++;
                            }
                            break;
                        case "lesk":
                            if (normalized) {
                                double maxSim = new Lesk(db).calcRelatednessOfWords(cpToken.getFeatures().get("string").toString(),
                                        cpToken.getFeatures().get("string").toString());

                                val = new Lesk(db).calcRelatednessOfWords(cpToken.getFeatures().get("string").toString(),
                                        rpToken.getFeatures().get("string").toString()) / maxSim;
                            } else {
                                val = new Lesk(db).calcRelatednessOfWords(cpToken.getFeatures().get("string").toString(),
                                        rpToken.getFeatures().get("string").toString());
                            }
                            if (!Double.isInfinite(val)) {
                                sum += val;
                                count++;
                            }
                            break;
                        case "lin":
                            if (normalized) {
                                double maxSim = new Lin(db).calcRelatednessOfWords(cpToken.getFeatures().get("string").toString(),
                                        cpToken.getFeatures().get("string").toString());

                                val = new Lin(db).calcRelatednessOfWords(cpToken.getFeatures().get("string").toString(),
                                        rpToken.getFeatures().get("string").toString()) / maxSim;
                            } else {
                                val = new Lin(db).calcRelatednessOfWords(cpToken.getFeatures().get("string").toString(),
                                        rpToken.getFeatures().get("string").toString());
                            }
                            if (!Double.isInfinite(val)) {
                                sum += val;
                                count++;
                            }
                            break;
                        case "path":
                            if (normalized) {
                                double maxSim = new Path(db).calcRelatednessOfWords(cpToken.getFeatures().get("string").toString(),
                                        cpToken.getFeatures().get("string").toString());

                                val = new Path(db).calcRelatednessOfWords(cpToken.getFeatures().get("string").toString(),
                                        rpToken.getFeatures().get("string").toString()) / maxSim;
                            } else {
                                val = new Path(db).calcRelatednessOfWords(cpToken.getFeatures().get("string").toString(),
                                        rpToken.getFeatures().get("string").toString());
                            }
                            if (!Double.isInfinite(val)) {
                                sum += val;
                                count++;
                            }
                            break;
                        case "resnik":
                            if (normalized) {
                                double maxSim = new Resnik(db).calcRelatednessOfWords(cpToken.getFeatures().get("string").toString(),
                                        cpToken.getFeatures().get("string").toString());

                                val = new Resnik(db).calcRelatednessOfWords(cpToken.getFeatures().get("string").toString(),
                                        rpToken.getFeatures().get("string").toString()) / maxSim;
                            } else {
                                val = new Resnik(db).calcRelatednessOfWords(cpToken.getFeatures().get("string").toString(),
                                        rpToken.getFeatures().get("string").toString());
                            }
                            if (!Double.isInfinite(val)) {
                                sum += val;
                                count++;
                            }
                            break;
                        case "wup":
                            if (normalized) {
                                double maxSim = new WuPalmer(db).calcRelatednessOfWords(cpToken.getFeatures().get("string").toString(),
                                        cpToken.getFeatures().get("string").toString());

                                val = new WuPalmer(db).calcRelatednessOfWords(cpToken.getFeatures().get("string").toString(),
                                        rpToken.getFeatures().get("string").toString()) / maxSim;
                            } else {
                                val = new WuPalmer(db).calcRelatednessOfWords(cpToken.getFeatures().get("string").toString(),
                                        rpToken.getFeatures().get("string").toString());
                            }
                            if (!Double.isInfinite(val)) {
                                sum += val;
                                count++;
                            }
                            break;
                    }
                } else {
                    sum++;
                }
            }
        }
        
        if (count > 0) {
            value.setValue(sum / count);
        } else {
            value.setValue(0d);
        }

        return value;
    }
}
