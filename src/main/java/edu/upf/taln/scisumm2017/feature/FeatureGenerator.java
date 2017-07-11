package edu.upf.taln.scisumm2017.feature;

import edu.upf.taln.ml.feat.*;
import edu.upf.taln.ml.feat.exception.FeatSetConsistencyException;
import edu.upf.taln.ml.feat.exception.FeatureException;
import edu.upf.taln.scisumm2017.feature.calculator.*;
import edu.upf.taln.scisumm2017.feature.context.DocumentCtx;
import edu.upf.taln.scisumm2017.reader.TrainingExample;
import gate.*;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import weka.core.converters.ArffSaver;

import java.io.File;
import java.io.IOException;
import java.util.*;

/**
 * Created by ahmed on 5/3/2016.
 */
public class FeatureGenerator {
    static HashMap<String, Document> RCDocuments;

    private static Set<String> matchClassValues = new HashSet<String>();
    private static Set<String> facetClassValues = new HashSet<String>();

    static {
        matchClassValues.add("MATCH");
        matchClassValues.add("NO_MATCH");
    }

    static {
        facetClassValues.add("AIM");
        facetClassValues.add("HYPOTHESIS");
        facetClassValues.add("METHOD");
        facetClassValues.add("RESULT");
        facetClassValues.add("IMPLICATION");
    }

    public static void generateMatchARFFfromFeaturesAnnotations(HashMap<String, Document> RCDocuments, String outputPath,
                                                                String rfolderName, boolean generateTraining) {

        String outputInstancesType = (generateTraining) ? "Match_Training" : "Match_Testing";
        String version = "1";

        Logger.getRootLogger().setLevel(Level.DEBUG);

        FeatureSet<TrainingExample, DocumentCtx> featSet = new FeatureSet<TrainingExample, DocumentCtx>();

        // Adding document identifier
        try {
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("SENTENCE_POSITION", new SentencePosition("sid")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("SENTENCE_SECTION_POSITION", new SentencePosition("ssid")));

            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("FACET_AIM", new SectionTitleFacet(new String[]{"aim", "objective", "purpose"})));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("FACET_HYPOTHESIS", new SectionTitleFacet(new String[]{"hypothesis", "possibility", "theory"})));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("FACET_IMPLICATION", new SectionTitleFacet(new String[]{"implication", "deduction", "entailment"})));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("FACET_METHOD", new SectionTitleFacet(new String[]{"method", "approach"})));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("FACET_RESULT", new SectionTitleFacet(new String[]{"result", "solution", "outcome", "answer", "evaluation"})));

            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("JIANGCONRATH_SIMILARITY", new WordNetSimilarity(true, "jiangconrath")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("LCH_SIMILARITY", new WordNetSimilarity(true, "lch")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("LESK_SIMILARITY", new WordNetSimilarity(true, "lesk")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("LIN_SIMILARITY", new WordNetSimilarity(true, "lin")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("PATH_SIMILARITY", new WordNetSimilarity(true, "path")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RESNIK_SIMILARITY", new WordNetSimilarity(true, "resnik")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("WUP_SIMILARITY", new WordNetSimilarity(true, "wup")));

            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("COSINE_SIMILARITY", new CosineSimilarity("LEMMA")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("BABELNET_COSINE_SIMILARITY", new CosineSimilarity("BABELNET")));

            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("PROBABILITY_APPROACH", new DrInventorFacetProbability("PROB_DRI_Approach")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("PROBABILITY_BACKGROUND", new DrInventorFacetProbability("PROB_DRI_Background")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("PROBABILITY_CHALLENGE", new DrInventorFacetProbability("PROB_DRI_Challenge")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("PROBABILITY_FUTUREWORK", new DrInventorFacetProbability("PROB_DRI_FutureWork")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("PROBABILITY_OUTCOME", new DrInventorFacetProbability("PROB_DRI_Outcome")));

            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CP_CITMARKER_COUNT", new CitationMarkerCount(true, "CP")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RP_CITMARKER_COUNT", new CitationMarkerCount(true, "RP")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CITMARKER_COUNT", new CitationMarkerCount(true, "BOTH")));

            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CP_CAUSEAFFECT_EXISTANCE", new CauseAffectExistance(true, "CP")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RP_CAUSEAFFECT_EXISTANCE", new CauseAffectExistance(true, "RP")));

            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CP_COREFCHAINS_COUNT", new CoRefChainsCount(true, "CP")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RP_COREFCHAINS_COUNT", new CoRefChainsCount(true, "RP")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("COREFCHAINS_COUNT", new CoRefChainsCount(true, "BOTH")));

            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZRESEARCHMT_PROP", new GazProbability("BOTH", "research")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZRESEARCHMT_PROP", new GazProbability("RP", "research")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZRESEARCHMT_PROP", new GazProbability("CP", "research")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZARGUMENTATIONMT_PROP", new GazProbability("BOTH", "argumentation")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZARGUMENTATIONMT_PROP", new GazProbability("RP", "argumentation")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZARGUMENTATIONMT_PROP", new GazProbability("CP", "argumentation")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZAWAREMT_PROP", new GazProbability("BOTH", "aware")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZAWAREMT_PROP", new GazProbability("RP", "aware")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZAWAREMT_PROP", new GazProbability("CP", "aware")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZUSEMT_PROP", new GazProbability("BOTH", "use")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZUSEMT_PROP", new GazProbability("RP", "use")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZUSEMT_PROP", new GazProbability("CP", "use")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZPROBLEMMT_PROP", new GazProbability("BOTH", "problem")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZPROBLEMMT_PROP", new GazProbability("RP", "problem")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZPROBLEMMT_PROP", new GazProbability("CP", "problem")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZSOLUTIONMT_PROP", new GazProbability("BOTH", "solution")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZSOLUTIONMT_PROP", new GazProbability("RP", "solution")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZSOLUTIONMT_PROP", new GazProbability("CP", "solution")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZBETTERSOLUTIONMT_PROP", new GazProbability("BOTH", "better_solution")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZBETTERSOLUTIONMT_PROP", new GazProbability("RP", "better_solution")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZBETTERSOLUTIONMT_PROP", new GazProbability("CP", "better_solution")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZTEXTSTRUCTUREMT_PROP", new GazProbability("BOTH", "textstructure")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZTEXTSTRUCTUREMT_PROP", new GazProbability("RP", "textstructure")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZTEXTSTRUCTUREMT_PROP", new GazProbability("CP", "textstructure")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZINTRESTMT_PROP", new GazProbability("BOTH", "interest")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZINTRESTMT_PROP", new GazProbability("RP", "interest")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZINTRESTMT_PROP", new GazProbability("CP", "interest")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZCONTINUEMT_PROP", new GazProbability("BOTH", "continue")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZCONTINUEMT_PROP", new GazProbability("RP", "continue")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZCONTINUEMT_PROP", new GazProbability("CP", "continue")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZFUTUREINTERESTMT_PROP", new GazProbability("BOTH", "future_interest")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZFUTUREINTERESTMT_PROP", new GazProbability("RP", "future_interest")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZFUTUREINTERESTMT_PROP", new GazProbability("CP", "future_interest")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZNEEDMT_PROP", new GazProbability("BOTH", "need")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZNEEDMT_PROP", new GazProbability("RP", "need")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZNEEDMT_PROP", new GazProbability("CP", "need")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZAFFECTMT_PROP", new GazProbability("BOTH", "affect")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZAFFECTMT_PROP", new GazProbability("RP", "affect")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZAFFECTMT_PROP", new GazProbability("CP", "affect")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZPRESENTATIONMT_PROP", new GazProbability("BOTH", "presentation")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZPRESENTATIONMT_PROP", new GazProbability("RP", "presentation")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZPRESENTATIONMT_PROP", new GazProbability("CP", "presentation")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZCONTRASTMT_PROP", new GazProbability("BOTH", "contrast")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZCONTRASTMT_PROP", new GazProbability("RP", "contrast")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZCONTRASTMT_PROP", new GazProbability("CP", "contrast")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZCHANGEMT_PROP", new GazProbability("BOTH", "change")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZCHANGEMT_PROP", new GazProbability("RP", "change")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZCHANGEMT_PROP", new GazProbability("CP", "change")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZCOMPARISONMT_PROP", new GazProbability("BOTH", "comparison")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZCOMPARISONMT_PROP", new GazProbability("RP", "comparison")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZCOMPARISONMT_PROP", new GazProbability("CP", "comparison")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZSIMILARMT_PROP", new GazProbability("BOTH", "similar")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZSIMILARMT_PROP", new GazProbability("RP", "similar")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZSIMILARMT_PROP", new GazProbability("CP", "similar")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZCOMPARISONADJMT_PROP", new GazProbability("BOTH", "comparison_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZCOMPARISONADJMT_PROP", new GazProbability("RP", "comparison_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZCOMPARISONADJMT_PROP", new GazProbability("CP", "comparison_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZFUTUREADJMT_PROP", new GazProbability("BOTH", "future_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZFUTUREADJMT_PROP", new GazProbability("RP", "future_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZFUTUREADJMT_PROP", new GazProbability("CP", "future_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZINTERESTNOUNMT_PROP", new GazProbability("BOTH", "interest_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZINTERESTNOUNMT_PROP", new GazProbability("RP", "interest_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZINTERESTNOUNMT_PROP", new GazProbability("CP", "interest_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZQUESTIONNOUNMT_PROP", new GazProbability("BOTH", "question_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZQUESTIONNOUNMT_PROP", new GazProbability("RP", "question_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZQUESTIONNOUNMT_PROP", new GazProbability("CP", "question_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZAWAREADJMT_PROP", new GazProbability("BOTH", "aware_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZAWAREADJMT_PROP", new GazProbability("RP", "aware_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZAWAREADJMT_PROP", new GazProbability("CP", "aware_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZARGUMENTATIONNOUNMT_PROP", new GazProbability("BOTH", "argumentation_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZARGUMENTATIONNOUNMT_PROP", new GazProbability("RP", "argumentation_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZARGUMENTATIONNOUNMT_PROP", new GazProbability("CP", "argumentation_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZSIMILARNOUNMT_PROP", new GazProbability("BOTH", "similar_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZSIMILARNOUNMT_PROP", new GazProbability("RP", "similar_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZSIMILARNOUNMT_PROP", new GazProbability("CP", "similar_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZEARLIERADJMT_PROP", new GazProbability("BOTH", "earlier_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZEARLIERADJMT_PROP", new GazProbability("RP", "earlier_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZEARLIERADJMT_PROP", new GazProbability("CP", "earlier_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZRESEARCHADJMT_PROP", new GazProbability("BOTH", "research_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZRESEARCHADJMT_PROP", new GazProbability("RP", "research_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZRESEARCHADJMT_PROP", new GazProbability("CP", "research_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZNEEDADJMT_PROP", new GazProbability("BOTH", "need_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZNEEDADJMT_PROP", new GazProbability("RP", "need_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZNEEDADJMT_PROP", new GazProbability("CP", "need_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZREFERENTIALMT_PROP", new GazProbability("BOTH", "referential")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZREFERENTIALMT_PROP", new GazProbability("RP", "referential")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZREFERENTIALMT_PROP", new GazProbability("CP", "referential")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZQUESTIONMT_PROP", new GazProbability("BOTH", "question")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZQUESTIONMT_PROP", new GazProbability("RP", "question")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZQUESTIONMT_PROP", new GazProbability("CP", "question")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZWORKNOUNMT_PROP", new GazProbability("BOTH", "work_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZWORKNOUNMT_PROP", new GazProbability("RP", "work_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZWORKNOUNMT_PROP", new GazProbability("CP", "work_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZCHANGEADJMT_PROP", new GazProbability("BOTH", "change_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZCHANGEADJMT_PROP", new GazProbability("RP", "change_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZCHANGEADJMT_PROP", new GazProbability("CP", "change_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZDISCIPLINEMT_PROP", new GazProbability("BOTH", "discipline")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZDISCIPLINEMT_PROP", new GazProbability("RP", "discipline")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZDISCIPLINEMT_PROP", new GazProbability("CP", "discipline")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZGIVENMT_PROP", new GazProbability("BOTH", "given")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZGIVENMT_PROP", new GazProbability("RP", "given")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZGIVENMT_PROP", new GazProbability("CP", "given")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZBADADJMT_PROP", new GazProbability("BOTH", "bad_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZBADADJMT_PROP", new GazProbability("RP", "bad_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZBADADJMT_PROP", new GazProbability("CP", "bad_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZCONTRASTNOUNMT_PROP", new GazProbability("BOTH", "contrast_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZCONTRASTNOUNMT_PROP", new GazProbability("RP", "contrast_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZCONTRASTNOUNMT_PROP", new GazProbability("CP", "contrast_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZNEEDNOUNMT_PROP", new GazProbability("BOTH", "need_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZNEEDNOUNMT_PROP", new GazProbability("RP", "need_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZNEEDNOUNMT_PROP", new GazProbability("CP", "need_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZAIMNOUNMT_PROP", new GazProbability("BOTH", "aim_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZAIMNOUNMT_PROP", new GazProbability("RP", "aim_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZAIMNOUNMT_PROP", new GazProbability("CP", "aim_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZCONTRASTADJMT_PROP", new GazProbability("BOTH", "contrast_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZCONTRASTADJMT_PROP", new GazProbability("RP", "contrast_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZCONTRASTADJMT_PROP", new GazProbability("CP", "contrast_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZSOLUTIONNOUNMT_PROP", new GazProbability("BOTH", "solution_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZSOLUTIONNOUNMT_PROP", new GazProbability("RP", "solution_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZSOLUTIONNOUNMT_PROP", new GazProbability("CP", "solution_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZTRADITIONNOUNMT_PROP", new GazProbability("BOTH", "tradition_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZTRADITIONNOUNMT_PROP", new GazProbability("RP", "tradition_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZTRADITIONNOUNMT_PROP", new GazProbability("CP", "tradition_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZFIRSTPRONMT_PROP", new GazProbability("BOTH", "first_pron")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZFIRSTPRONMT_PROP", new GazProbability("RP", "first_pron")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZFIRSTPRONMT_PROP", new GazProbability("CP", "first_pron")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZPROFESSIONALSMT_PROP", new GazProbability("BOTH", "professionals")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZPROFESSIONALSMT_PROP", new GazProbability("RP", "professionals")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZPROFESSIONALSMT_PROP", new GazProbability("CP", "professionals")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZPROBLEMNOUNMT_PROP", new GazProbability("BOTH", "problem_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZPROBLEMNOUNMT_PROP", new GazProbability("RP", "problem_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZPROBLEMNOUNMT_PROP", new GazProbability("CP", "problem_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZNEGATIONMT_PROP", new GazProbability("BOTH", "negation")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZNEGATIONMT_PROP", new GazProbability("RP", "negation")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZNEGATIONMT_PROP", new GazProbability("CP", "negation")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZTEXTNOUNMT_PROP", new GazProbability("BOTH", "text_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZTEXTNOUNMT_PROP", new GazProbability("RP", "text_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZTEXTNOUNMT_PROP", new GazProbability("CP", "text_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZPROBLEMADJMT_PROP", new GazProbability("BOTH", "problem_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZPROBLEMADJMT_PROP", new GazProbability("RP", "problem_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZPROBLEMADJMT_PROP", new GazProbability("CP", "problem_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZTHIRDPRONMT_PROP", new GazProbability("BOTH", "third_pron")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZTHIRDPRONMT_PROP", new GazProbability("RP", "third_pron")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZTHIRDPRONMT_PROP", new GazProbability("CP", "third_pron")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZTRADITIONADJMT_PROP", new GazProbability("BOTH", "tradition_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZTRADITIONADJMT_PROP", new GazProbability("RP", "tradition_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZTRADITIONADJMT_PROP", new GazProbability("CP", "tradition_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZPRESENTATIONNOUNMT_PROP", new GazProbability("BOTH", "presentation_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZPRESENTATIONNOUNMT_PROP", new GazProbability("RP", "presentation_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZPRESENTATIONNOUNMT_PROP", new GazProbability("CP", "presentation_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZRESEARCHNOUNMT_PROP", new GazProbability("BOTH", "research_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZRESEARCHNOUNMT_PROP", new GazProbability("RP", "research_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZRESEARCHNOUNMT_PROP", new GazProbability("CP", "research_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZMAINADJMT_PROP", new GazProbability("BOTH", "main_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZMAINADJMT_PROP", new GazProbability("RP", "main_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZMAINADJMT_PROP", new GazProbability("CP", "main_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZREFLEXSIVEMT_PROP", new GazProbability("BOTH", "reflexive")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZREFLEXSIVEMT_PROP", new GazProbability("RP", "reflexive")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZREFLEXSIVEMT_PROP", new GazProbability("CP", "reflexive")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZNEDADJMT_PROP", new GazProbability("BOTH", "ned_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZNEDADJMT_PROP", new GazProbability("RP", "ned_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZNEDADJMT_PROP", new GazProbability("CP", "ned_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZMANYMT_PROP", new GazProbability("BOTH", "many")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZMANYMT_PROP", new GazProbability("RP", "many")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZMANYMT_PROP", new GazProbability("CP", "many")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZCOMPARISONNOUNMT_PROP", new GazProbability("BOTH", "comparison_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZCOMPARISONNOUNMT_PROP", new GazProbability("RP", "comparison_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZCOMPARISONNOUNMT_PROP", new GazProbability("CP", "comparison_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZGOODADJMT_PROP", new GazProbability("BOTH", "good_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZGOODADJMT_PROP", new GazProbability("RP", "good_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZGOODADJMT_PROP", new GazProbability("CP", "good_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZCHANGENOUNMT_PROP", new GazProbability("BOTH", "change_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZCHANGENOUNMT_PROP", new GazProbability("RP", "change_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZCHANGENOUNMT_PROP", new GazProbability("CP", "change_noun")));
/*
            featSet.addFeature(new StringW<TrainingExample, DocumentCtx>("RP_SENTENCEBIGRAMLEMMAS_STRING", new SentenceNGramsStrings("RP", "LemmasNGrams", "2-gram")));
            featSet.addFeature(new StringW<TrainingExample, DocumentCtx>("CP_SENTENCEBIGRAMLEMMAS_STRING", new SentenceNGramsStrings("CP", "LemmasNGrams", "2-gram")));
            featSet.addFeature(new StringW<TrainingExample, DocumentCtx>("RP_SENTENCEBIGRAMPOSS_STRING", new SentenceNGramsStrings("RP", "POSNGrams", "2-gram")));
            featSet.addFeature(new StringW<TrainingExample, DocumentCtx>("CP_SENTENCEBIGRAMPOSS_STRING", new SentenceNGramsStrings("CP", "POSNGrams", "2-gram")));

            featSet.addFeature(new StringW<TrainingExample, DocumentCtx>("RP_SENTENCEPOSS_STRING", new SentenceNGramsStrings("RP", "POSNGrams", "1-gram")));
            featSet.addFeature(new StringW<TrainingExample, DocumentCtx>("CP_SENTENCEPOSS_STRING", new SentenceNGramsStrings("CP", "POSNGrams", "1-gram")));
            featSet.addFeature(new StringW<TrainingExample, DocumentCtx>("RP_SENTENCELEMMAS_STRING", new SentenceNGramsStrings("RP", "LemmasNGrams", "1-gram")));
            featSet.addFeature(new StringW<TrainingExample, DocumentCtx>("CP_SENTENCELEMMAS_STRING", new SentenceNGramsStrings("CP", "LemmasNGrams", "1-gram")));*/

            // Class feature (lasts)
            featSet.addFeature(new NominalW<TrainingExample, DocumentCtx>("class", matchClassValues, new ClassGetter(true)));

        } catch (FeatureException e) {
            System.out.println("Error instantiating feature generation template.");
            e.printStackTrace();
            return;
        } catch (IOException e) {
            e.printStackTrace();
        }

        System.out.println("ARFF File - " + outputInstancesType + " " + rfolderName + " instances generation...");

        Document rp = RCDocuments.get(rfolderName);
        AnnotationSet rpMatch_Features = rp.getAnnotations("Match_Features");
        AnnotationSet rpNoMatch_Features = rp.getAnnotations("NO_Match_Features");

        Iterator rpMatchIterator = rpMatch_Features.iterator();
        Iterator rpNoMatchIterator = rpNoMatch_Features.iterator();

        while (rpMatchIterator.hasNext())
        {
            Annotation rpMatch = (Annotation) rpMatchIterator.next();
            DocumentCtx trCtx = new DocumentCtx(rp, null);
            TrainingExample te = new TrainingExample(rpMatch, null, 1);
            featSet.addElement(te, trCtx);
        }

        while (rpNoMatchIterator.hasNext())
        {
            Annotation rpMatch = (Annotation) rpNoMatchIterator.next();
            DocumentCtx trCtx = new DocumentCtx(rp, null);
            TrainingExample te = new TrainingExample(rpMatch, null, 0);
            featSet.addElement(te, trCtx);
        }

        // --- STORE ARFF:
        System.out.println("STORING ARFF... " + rfolderName);
        try {
            ArffSaver saver = new ArffSaver();
            saver.setInstances(FeatUtil.wekaInstanceGeneration(featSet, rfolderName + " " +
                    outputInstancesType + " scisumm2017_v_" + version));
            saver.setFile(new File(outputPath + File.separator + "scisumm2017_" + outputInstancesType + "_v_" + version + ".arff"));
            saver.writeBatch();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (FeatSetConsistencyException e) {
            e.printStackTrace();
        }

    }

    public static void generateFacetARFFfromFeaturesAnnotations(HashMap<String, Document> RCDocuments, String outputPath,
                                                                String rfolderName, boolean generateTraining) {
        String outputInstancesType = (generateTraining) ? "Facet_Training" : "Facet_Testing";
        String version = "1";

        Logger.getRootLogger().setLevel(Level.DEBUG);

        FeatureSet<TrainingExample, DocumentCtx> featSet = new FeatureSet<TrainingExample, DocumentCtx>();

        // Adding document identifier
        try {
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("SENTENCE_POSITION", new SentencePosition("sid")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("SENTENCE_SECTION_POSITION", new SentencePosition("ssid")));

            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("FACET_AIM", new SectionTitleFacet(new String[]{"aim", "objective", "purpose"})));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("FACET_HYPOTHESIS", new SectionTitleFacet(new String[]{"hypothesis", "possibility", "theory"})));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("FACET_IMPLICATION", new SectionTitleFacet(new String[]{"implication", "deduction", "entailment"})));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("FACET_METHOD", new SectionTitleFacet(new String[]{"method", "approach"})));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("FACET_RESULT", new SectionTitleFacet(new String[]{"result", "solution", "outcome", "answer", "evaluation"})));

            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("JIANGCONRATH_SIMILARITY", new WordNetSimilarity(true, "jiangconrath")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("LCH_SIMILARITY", new WordNetSimilarity(true, "lch")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("LESK_SIMILARITY", new WordNetSimilarity(true, "lesk")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("LIN_SIMILARITY", new WordNetSimilarity(true, "lin")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("PATH_SIMILARITY", new WordNetSimilarity(true, "path")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RESNIK_SIMILARITY", new WordNetSimilarity(true, "resnik")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("WUP_SIMILARITY", new WordNetSimilarity(true, "wup")));

            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("COSINE_SIMILARITY", new CosineSimilarity("LEMMA")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("BABELNET_COSINE_SIMILARITY", new CosineSimilarity("BABELNET")));

            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("PROBABILITY_APPROACH", new DrInventorFacetProbability("PROB_DRI_Approach")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("PROBABILITY_BACKGROUND", new DrInventorFacetProbability("PROB_DRI_Background")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("PROBABILITY_CHALLENGE", new DrInventorFacetProbability("PROB_DRI_Challenge")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("PROBABILITY_FUTUREWORK", new DrInventorFacetProbability("PROB_DRI_FutureWork")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("PROBABILITY_OUTCOME", new DrInventorFacetProbability("PROB_DRI_Outcome")));

            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CP_CITMARKER_COUNT", new CitationMarkerCount(true, "CP")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RP_CITMARKER_COUNT", new CitationMarkerCount(true, "RP")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CITMARKER_COUNT", new CitationMarkerCount(true, "BOTH")));

            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CP_CAUSEAFFECT_EXISTANCE", new CauseAffectExistance(true, "CP")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RP_CAUSEAFFECT_EXISTANCE", new CauseAffectExistance(true, "RP")));

            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CP_COREFCHAINS_COUNT", new CoRefChainsCount(true, "CP")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RP_COREFCHAINS_COUNT", new CoRefChainsCount(true, "RP")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("COREFCHAINS_COUNT", new CoRefChainsCount(true, "BOTH")));

            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZRESEARCHMT_PROP", new GazProbability("BOTH", "research")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZRESEARCHMT_PROP", new GazProbability("RP", "research")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZRESEARCHMT_PROP", new GazProbability("CP", "research")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZARGUMENTATIONMT_PROP", new GazProbability("BOTH", "argumentation")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZARGUMENTATIONMT_PROP", new GazProbability("RP", "argumentation")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZARGUMENTATIONMT_PROP", new GazProbability("CP", "argumentation")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZAWAREMT_PROP", new GazProbability("BOTH", "aware")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZAWAREMT_PROP", new GazProbability("RP", "aware")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZAWAREMT_PROP", new GazProbability("CP", "aware")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZUSEMT_PROP", new GazProbability("BOTH", "use")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZUSEMT_PROP", new GazProbability("RP", "use")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZUSEMT_PROP", new GazProbability("CP", "use")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZPROBLEMMT_PROP", new GazProbability("BOTH", "problem")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZPROBLEMMT_PROP", new GazProbability("RP", "problem")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZPROBLEMMT_PROP", new GazProbability("CP", "problem")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZSOLUTIONMT_PROP", new GazProbability("BOTH", "solution")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZSOLUTIONMT_PROP", new GazProbability("RP", "solution")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZSOLUTIONMT_PROP", new GazProbability("CP", "solution")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZBETTERSOLUTIONMT_PROP", new GazProbability("BOTH", "better_solution")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZBETTERSOLUTIONMT_PROP", new GazProbability("RP", "better_solution")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZBETTERSOLUTIONMT_PROP", new GazProbability("CP", "better_solution")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZTEXTSTRUCTUREMT_PROP", new GazProbability("BOTH", "textstructure")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZTEXTSTRUCTUREMT_PROP", new GazProbability("RP", "textstructure")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZTEXTSTRUCTUREMT_PROP", new GazProbability("CP", "textstructure")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZINTRESTMT_PROP", new GazProbability("BOTH", "interest")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZINTRESTMT_PROP", new GazProbability("RP", "interest")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZINTRESTMT_PROP", new GazProbability("CP", "interest")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZCONTINUEMT_PROP", new GazProbability("BOTH", "continue")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZCONTINUEMT_PROP", new GazProbability("RP", "continue")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZCONTINUEMT_PROP", new GazProbability("CP", "continue")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZFUTUREINTERESTMT_PROP", new GazProbability("BOTH", "future_interest")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZFUTUREINTERESTMT_PROP", new GazProbability("RP", "future_interest")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZFUTUREINTERESTMT_PROP", new GazProbability("CP", "future_interest")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZNEEDMT_PROP", new GazProbability("BOTH", "need")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZNEEDMT_PROP", new GazProbability("RP", "need")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZNEEDMT_PROP", new GazProbability("CP", "need")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZAFFECTMT_PROP", new GazProbability("BOTH", "affect")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZAFFECTMT_PROP", new GazProbability("RP", "affect")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZAFFECTMT_PROP", new GazProbability("CP", "affect")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZPRESENTATIONMT_PROP", new GazProbability("BOTH", "presentation")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZPRESENTATIONMT_PROP", new GazProbability("RP", "presentation")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZPRESENTATIONMT_PROP", new GazProbability("CP", "presentation")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZCONTRASTMT_PROP", new GazProbability("BOTH", "contrast")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZCONTRASTMT_PROP", new GazProbability("RP", "contrast")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZCONTRASTMT_PROP", new GazProbability("CP", "contrast")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZCHANGEMT_PROP", new GazProbability("BOTH", "change")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZCHANGEMT_PROP", new GazProbability("RP", "change")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZCHANGEMT_PROP", new GazProbability("CP", "change")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZCOMPARISONMT_PROP", new GazProbability("BOTH", "comparison")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZCOMPARISONMT_PROP", new GazProbability("RP", "comparison")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZCOMPARISONMT_PROP", new GazProbability("CP", "comparison")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZSIMILARMT_PROP", new GazProbability("BOTH", "similar")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZSIMILARMT_PROP", new GazProbability("RP", "similar")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZSIMILARMT_PROP", new GazProbability("CP", "similar")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZCOMPARISONADJMT_PROP", new GazProbability("BOTH", "comparison_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZCOMPARISONADJMT_PROP", new GazProbability("RP", "comparison_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZCOMPARISONADJMT_PROP", new GazProbability("CP", "comparison_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZFUTUREADJMT_PROP", new GazProbability("BOTH", "future_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZFUTUREADJMT_PROP", new GazProbability("RP", "future_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZFUTUREADJMT_PROP", new GazProbability("CP", "future_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZINTERESTNOUNMT_PROP", new GazProbability("BOTH", "interest_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZINTERESTNOUNMT_PROP", new GazProbability("RP", "interest_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZINTERESTNOUNMT_PROP", new GazProbability("CP", "interest_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZQUESTIONNOUNMT_PROP", new GazProbability("BOTH", "question_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZQUESTIONNOUNMT_PROP", new GazProbability("RP", "question_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZQUESTIONNOUNMT_PROP", new GazProbability("CP", "question_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZAWAREADJMT_PROP", new GazProbability("BOTH", "aware_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZAWAREADJMT_PROP", new GazProbability("RP", "aware_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZAWAREADJMT_PROP", new GazProbability("CP", "aware_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZARGUMENTATIONNOUNMT_PROP", new GazProbability("BOTH", "argumentation_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZARGUMENTATIONNOUNMT_PROP", new GazProbability("RP", "argumentation_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZARGUMENTATIONNOUNMT_PROP", new GazProbability("CP", "argumentation_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZSIMILARNOUNMT_PROP", new GazProbability("BOTH", "similar_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZSIMILARNOUNMT_PROP", new GazProbability("RP", "similar_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZSIMILARNOUNMT_PROP", new GazProbability("CP", "similar_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZEARLIERADJMT_PROP", new GazProbability("BOTH", "earlier_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZEARLIERADJMT_PROP", new GazProbability("RP", "earlier_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZEARLIERADJMT_PROP", new GazProbability("CP", "earlier_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZRESEARCHADJMT_PROP", new GazProbability("BOTH", "research_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZRESEARCHADJMT_PROP", new GazProbability("RP", "research_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZRESEARCHADJMT_PROP", new GazProbability("CP", "research_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZNEEDADJMT_PROP", new GazProbability("BOTH", "need_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZNEEDADJMT_PROP", new GazProbability("RP", "need_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZNEEDADJMT_PROP", new GazProbability("CP", "need_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZREFERENTIALMT_PROP", new GazProbability("BOTH", "referential")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZREFERENTIALMT_PROP", new GazProbability("RP", "referential")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZREFERENTIALMT_PROP", new GazProbability("CP", "referential")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZQUESTIONMT_PROP", new GazProbability("BOTH", "question")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZQUESTIONMT_PROP", new GazProbability("RP", "question")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZQUESTIONMT_PROP", new GazProbability("CP", "question")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZWORKNOUNMT_PROP", new GazProbability("BOTH", "work_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZWORKNOUNMT_PROP", new GazProbability("RP", "work_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZWORKNOUNMT_PROP", new GazProbability("CP", "work_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZCHANGEADJMT_PROP", new GazProbability("BOTH", "change_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZCHANGEADJMT_PROP", new GazProbability("RP", "change_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZCHANGEADJMT_PROP", new GazProbability("CP", "change_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZDISCIPLINEMT_PROP", new GazProbability("BOTH", "discipline")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZDISCIPLINEMT_PROP", new GazProbability("RP", "discipline")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZDISCIPLINEMT_PROP", new GazProbability("CP", "discipline")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZGIVENMT_PROP", new GazProbability("BOTH", "given")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZGIVENMT_PROP", new GazProbability("RP", "given")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZGIVENMT_PROP", new GazProbability("CP", "given")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZBADADJMT_PROP", new GazProbability("BOTH", "bad_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZBADADJMT_PROP", new GazProbability("RP", "bad_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZBADADJMT_PROP", new GazProbability("CP", "bad_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZCONTRASTNOUNMT_PROP", new GazProbability("BOTH", "contrast_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZCONTRASTNOUNMT_PROP", new GazProbability("RP", "contrast_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZCONTRASTNOUNMT_PROP", new GazProbability("CP", "contrast_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZNEEDNOUNMT_PROP", new GazProbability("BOTH", "need_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZNEEDNOUNMT_PROP", new GazProbability("RP", "need_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZNEEDNOUNMT_PROP", new GazProbability("CP", "need_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZAIMNOUNMT_PROP", new GazProbability("BOTH", "aim_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZAIMNOUNMT_PROP", new GazProbability("RP", "aim_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZAIMNOUNMT_PROP", new GazProbability("CP", "aim_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZCONTRASTADJMT_PROP", new GazProbability("BOTH", "contrast_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZCONTRASTADJMT_PROP", new GazProbability("RP", "contrast_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZCONTRASTADJMT_PROP", new GazProbability("CP", "contrast_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZSOLUTIONNOUNMT_PROP", new GazProbability("BOTH", "solution_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZSOLUTIONNOUNMT_PROP", new GazProbability("RP", "solution_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZSOLUTIONNOUNMT_PROP", new GazProbability("CP", "solution_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZTRADITIONNOUNMT_PROP", new GazProbability("BOTH", "tradition_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZTRADITIONNOUNMT_PROP", new GazProbability("RP", "tradition_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZTRADITIONNOUNMT_PROP", new GazProbability("CP", "tradition_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZFIRSTPRONMT_PROP", new GazProbability("BOTH", "first_pron")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZFIRSTPRONMT_PROP", new GazProbability("RP", "first_pron")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZFIRSTPRONMT_PROP", new GazProbability("CP", "first_pron")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZPROFESSIONALSMT_PROP", new GazProbability("BOTH", "professionals")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZPROFESSIONALSMT_PROP", new GazProbability("RP", "professionals")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZPROFESSIONALSMT_PROP", new GazProbability("CP", "professionals")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZPROBLEMNOUNMT_PROP", new GazProbability("BOTH", "problem_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZPROBLEMNOUNMT_PROP", new GazProbability("RP", "problem_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZPROBLEMNOUNMT_PROP", new GazProbability("CP", "problem_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZNEGATIONMT_PROP", new GazProbability("BOTH", "negation")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZNEGATIONMT_PROP", new GazProbability("RP", "negation")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZNEGATIONMT_PROP", new GazProbability("CP", "negation")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZTEXTNOUNMT_PROP", new GazProbability("BOTH", "text_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZTEXTNOUNMT_PROP", new GazProbability("RP", "text_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZTEXTNOUNMT_PROP", new GazProbability("CP", "text_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZPROBLEMADJMT_PROP", new GazProbability("BOTH", "problem_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZPROBLEMADJMT_PROP", new GazProbability("RP", "problem_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZPROBLEMADJMT_PROP", new GazProbability("CP", "problem_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZTHIRDPRONMT_PROP", new GazProbability("BOTH", "third_pron")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZTHIRDPRONMT_PROP", new GazProbability("RP", "third_pron")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZTHIRDPRONMT_PROP", new GazProbability("CP", "third_pron")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZTRADITIONADJMT_PROP", new GazProbability("BOTH", "tradition_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZTRADITIONADJMT_PROP", new GazProbability("RP", "tradition_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZTRADITIONADJMT_PROP", new GazProbability("CP", "tradition_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZPRESENTATIONNOUNMT_PROP", new GazProbability("BOTH", "presentation_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZPRESENTATIONNOUNMT_PROP", new GazProbability("RP", "presentation_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZPRESENTATIONNOUNMT_PROP", new GazProbability("CP", "presentation_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZRESEARCHNOUNMT_PROP", new GazProbability("BOTH", "research_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZRESEARCHNOUNMT_PROP", new GazProbability("RP", "research_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZRESEARCHNOUNMT_PROP", new GazProbability("CP", "research_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZMAINADJMT_PROP", new GazProbability("BOTH", "main_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZMAINADJMT_PROP", new GazProbability("RP", "main_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZMAINADJMT_PROP", new GazProbability("CP", "main_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZREFLEXSIVEMT_PROP", new GazProbability("BOTH", "reflexive")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZREFLEXSIVEMT_PROP", new GazProbability("RP", "reflexive")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZREFLEXSIVEMT_PROP", new GazProbability("CP", "reflexive")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZNEDADJMT_PROP", new GazProbability("BOTH", "ned_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZNEDADJMT_PROP", new GazProbability("RP", "ned_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZNEDADJMT_PROP", new GazProbability("CP", "ned_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZMANYMT_PROP", new GazProbability("BOTH", "many")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZMANYMT_PROP", new GazProbability("RP", "many")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZMANYMT_PROP", new GazProbability("CP", "many")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZCOMPARISONNOUNMT_PROP", new GazProbability("BOTH", "comparison_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZCOMPARISONNOUNMT_PROP", new GazProbability("RP", "comparison_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZCOMPARISONNOUNMT_PROP", new GazProbability("CP", "comparison_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZGOODADJMT_PROP", new GazProbability("BOTH", "good_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZGOODADJMT_PROP", new GazProbability("RP", "good_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZGOODADJMT_PROP", new GazProbability("CP", "good_adj")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("GAZCHANGENOUNMT_PROP", new GazProbability("BOTH", "change_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("RPGAZCHANGENOUNMT_PROP", new GazProbability("RP", "change_noun")));
            featSet.addFeature(new NumericW<TrainingExample, DocumentCtx>("CPGAZCHANGENOUNMT_PROP", new GazProbability("CP", "change_noun")));

            featSet.addFeature(new StringW<TrainingExample, DocumentCtx>("SENTENCEBIGRAMLEMMAS_STRING", new SentenceNGramsStrings("BOTH", "LemmasNGrams", "2-gram")));
            featSet.addFeature(new StringW<TrainingExample, DocumentCtx>("SENTENCELEMMAS_STRING", new SentenceNGramsStrings("BOTH", "LemmasNGrams", "1-gram")));
            featSet.addFeature(new StringW<TrainingExample, DocumentCtx>("SENTENCEBIGRAMPOSS_STRING", new SentenceNGramsStrings("BOTH", "POSNGrams", "2-gram")));
            featSet.addFeature(new StringW<TrainingExample, DocumentCtx>("SENTENCEPOSS_STRING", new SentenceNGramsStrings("BOTH", "POSNGrams", "1-gram")));

            // Class feature (lasts)
            featSet.addFeature(new NominalW<TrainingExample, DocumentCtx>("class", facetClassValues, new ClassGetter(false)));
        } catch (FeatureException e) {
            System.out.println("Error instantiating feature generation template.");
            e.printStackTrace();
            return;
        } catch (IOException e) {
            e.printStackTrace();
        }

        System.out.println("ARFF File - " + outputInstancesType + " " + rfolderName + " instances generation...");

        Document rp = RCDocuments.get(rfolderName);
        AnnotationSet rpFacet_Features = rp.getAnnotations("Facet_Features");

        Iterator rpFacetIterator = rpFacet_Features.iterator();
        while (rpFacetIterator.hasNext())
        {
            Annotation rpFacet = (Annotation) rpFacetIterator.next();
            DocumentCtx trCtx = new DocumentCtx(rp, null);
            TrainingExample te = new TrainingExample(rpFacet, null, rpFacet.getFeatures().get("class").toString());
            featSet.addElement(te, trCtx);
        }

        // --- STORE ARFF:
        System.out.println("STORING ARFF... " + rfolderName);
        try {
            ArffSaver saver = new ArffSaver();
            saver.setInstances(FeatUtil.wekaInstanceGeneration(featSet, rfolderName + " " +
                    outputInstancesType + " scisumm2017_v_" + version));
            saver.setFile(new File(outputPath + File.separator + "scisumm2017_" + outputInstancesType + "_v_" + version + ".arff"));
            saver.writeBatch();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (FeatSetConsistencyException e) {
            e.printStackTrace();
        }


    }

}
