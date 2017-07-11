package edu.upf.taln.scisumm2017.feature.calculator;

import edu.upf.taln.ml.feat.base.FeatCalculator;
import edu.upf.taln.ml.feat.base.MyDouble;
import edu.upf.taln.scisumm2017.feature.context.DocumentCtx;
import edu.upf.taln.scisumm2017.reader.TrainingExample;
import gate.Annotation;
import gate.AnnotationSet;
import gate.Document;
import gate.FeatureMap;

/**
 * Created by ahmed on 7/8/16.
 */
public class GazProbability implements FeatCalculator<Double, TrainingExample, DocumentCtx> {
    String target;
    String gazListName;

    public GazProbability(String target, String gazListName) {
        this.target = target;
        this.gazListName = gazListName;
    }

    @Override
    public MyDouble calculateFeature(TrainingExample obj, DocumentCtx docs, String GazProbability) {
        MyDouble value = new MyDouble(0d);
        int totalCount = 0;
        int MTCount = 0;

        try
        {
            Document rp = docs.getReferenceDoc();
            Document cp = docs.getCitationDoc();

            Annotation refSentence = obj.getReferenceSentence();
            Annotation citSentence = obj.getCitanceSentence();

            AnnotationSet rpLookups = rp.getAnnotations("Analysis").get("Lookup").get(refSentence.getStartNode().getOffset(),
                    refSentence.getEndNode().getOffset());
            AnnotationSet cpLookups = cp.getAnnotations("Analysis").get("Lookup").get(citSentence.getStartNode().getOffset(),
                    citSentence.getEndNode().getOffset());

            AnnotationSet rpTokens = rp.getAnnotations("Analysis").get("Token").get(refSentence.getStartNode().getOffset(),
                    refSentence.getEndNode().getOffset());
            AnnotationSet cpTokens = cp.getAnnotations("Analysis").get("Token").get(citSentence.getStartNode().getOffset(),
                    citSentence.getEndNode().getOffset());

            switch (target) {
                case "RP":
                    for (Annotation annotation : rpLookups) {
                        FeatureMap fm = annotation.getFeatures();
                        if (fm.containsValue(gazListName)) {
                            MTCount++;
                        }
                    }
                    totalCount = rpTokens.size();
                    break;
                case "CP":
                    for (Annotation annotation : cpLookups) {
                        FeatureMap fm = annotation.getFeatures();
                        if (fm.containsValue(gazListName)) {
                            MTCount++;
                        }
                    }
                    totalCount = cpTokens.size();
                    break;
                case "BOTH":
                    for (Annotation annotation : rpLookups) {
                        FeatureMap fm = annotation.getFeatures();
                        if (fm.containsValue(gazListName)) {
                            MTCount++;
                        }
                    }
                    for (Annotation annotation : cpLookups) {
                        FeatureMap fm = annotation.getFeatures();
                        if (fm.containsValue(gazListName)) {
                            MTCount++;
                        }
                    }
                    totalCount = rpTokens.size() + cpTokens.size();
                    break;
            }
            value.setValue(getPercentage(MTCount, totalCount));
        }
        catch (Exception e)
        {
            e.printStackTrace();
        }

        return value;
    }

    public static double getPercentage(int n, int total) {
        double proportion = 0d;
        if (total != 0) {
            proportion = ((double) n) / ((double) total);
        }
        return proportion;
    }
}
