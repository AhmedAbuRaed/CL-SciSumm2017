package edu.upf.taln.scisumm2017.feature.calculator;

import edu.upf.taln.ml.feat.base.FeatCalculator;
import edu.upf.taln.ml.feat.base.MyDouble;
import edu.upf.taln.scisumm2017.feature.context.DocumentCtx;
import edu.upf.taln.scisumm2017.reader.TrainingExample;
import gate.Annotation;
import gate.Document;

import java.util.regex.Pattern;

/**
 * Created by ahmed on 5/11/2016.
 */
public class SentencePosition implements FeatCalculator<Double, TrainingExample, DocumentCtx> {
    String positionID;

    public SentencePosition(String positionID)
    {
        this.positionID = positionID;
    }

    @Override
    public MyDouble calculateFeature(TrainingExample obj, DocumentCtx docs, String SentencePosition) {
        MyDouble value = new MyDouble(0d);
        Annotation refSentence = obj.getReferenceSentence();

        if (refSentence.getFeatures().containsKey(positionID)) {
            if (refSentence.getFeatures().get(positionID) != null && !refSentence.getFeatures().get(positionID).equals("")) {
                //VALIDATE String
                final String Digits = "(\\p{Digit}+)";
                final String HexDigits = "(\\p{XDigit}+)";
                // an exponent is 'e' or 'E' followed by an optionally
                // signed decimal integer.
                final String Exp = "[eE][+-]?" + Digits;
                final String fpRegex =
                        ("[\\x00-\\x20]*" +  // Optional leading "whitespace"
                                "[+-]?(" + // Optional sign character
                                "NaN|" +           // "NaN" string
                                "Infinity|" +      // "Infinity" string

                                // A decimal floating-point string representing a finite positive
                                // number without a leading sign has at most five basic pieces:
                                // Digits . Digits ExponentPart FloatTypeSuffix
                                //
                                // Since this method allows integer-only strings as input
                                // in addition to strings of floating-point literals, the
                                // two sub-patterns below are simplifications of the grammar
                                // productions from section 3.10.2 of
                                // The Java™ Language Specification.

                                // Digits ._opt Digits_opt ExponentPart_opt FloatTypeSuffix_opt
                                "(((" + Digits + "(\\.)?(" + Digits + "?)(" + Exp + ")?)|" +

                                // . Digits ExponentPart_opt FloatTypeSuffix_opt
                                "(\\.(" + Digits + ")(" + Exp + ")?)|" +

                                // Hexadecimal strings
                                "((" +
                                // 0[xX] HexDigits ._opt BinaryExponent FloatTypeSuffix_opt
                                "(0[xX]" + HexDigits + "(\\.)?)|" +

                                // 0[xX] HexDigits_opt . HexDigits BinaryExponent FloatTypeSuffix_opt
                                "(0[xX]" + HexDigits + "?(\\.)" + HexDigits + ")" +

                                ")[pP][+-]?" + Digits + "))" +
                                "[fFdD]?))" +
                                "[\\x00-\\x20]*");// Optional trailing "whitespace"


                if (Pattern.matches(fpRegex, (String) refSentence.getFeatures().get(positionID))) {
                    Double sentenceIDValue = Double.valueOf((String) refSentence.getFeatures().get(positionID)) + 1;
                    if (!sentenceIDValue.isInfinite() && !sentenceIDValue.isNaN() && sentenceIDValue > 0) {
                        value.setValue(1d / sentenceIDValue);
                    }
                }
            }
        }

        return value;
    }
}
