package edu.upf.taln.scisumm2017.feature.calculator;

import edu.upf.taln.ml.feat.base.FeatCalculator;
import edu.upf.taln.ml.feat.base.MyString;
import edu.upf.taln.scisumm2017.feature.context.DocumentCtx;
import edu.upf.taln.scisumm2017.reader.TrainingExample;

/**
 * Class: Match, NO_MATCH
 *
 * @author Francesco Ronzano
 */
public class ClassGetter implements FeatCalculator<String, TrainingExample, DocumentCtx> {

    private boolean matchClass;

    public ClassGetter(boolean matchClass) {
        this.matchClass = matchClass;
    }

    @Override
    public MyString calculateFeature(TrainingExample obj, DocumentCtx docs, String ClassGetter) {
        if (matchClass) {
            MyString Value = new MyString("NO_MATCH");

            if (obj != null && obj.getIsMatch() != null && obj.getIsMatch() >= 1) {
                Value.setValue("MATCH");
            }

            return Value;
        } else {
            MyString Value = new MyString(null);

            String facet = obj.getFacet();
            if (facet != null) {
                switch (facet) {
                    case "Aim_Citation":
                    case "Aim_Facet":
                        Value.setValue("Aim_Citation");
                        break;
                    case "Hypothesis_Citation":
                    case "Hypothesis_Facet":
                        Value.setValue("Hypothesis_Citation");
                        break;
                    case "Method_Citation":
                    case "Method_Facet":
                        Value.setValue("Method_Citation");
                        break;
                    case "Results_Citation":
                    case "Results_Facet":
                        Value.setValue("Results_Citation");
                        break;
                    case "Implication_Citation":
                    case "Implication_Facet":
                        Value.setValue("Implication_Citation");
                        break;
                }
            }

            return Value;
        }
    }
}
