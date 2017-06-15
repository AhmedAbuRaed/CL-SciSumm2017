package edu.upf.taln.scisumm2017.reader;

import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * Created by Ahmed on 4/3/17.
 */
public class BabelnetSynset {

    private TokenFragment tokenFragment;
    private CharFragment charFragment;
    private String babelSynsetID;
    @JsonProperty
    private String DBpediaURL;
    @JsonProperty
    private String BabelNetURL;
    private String score;
    private String coherenceScore;
    private String globalScore;
    private String source;

    public TokenFragment getTokenFragment() {
        return tokenFragment;
    }

    public void setTokenFragment(TokenFragment tokenFragment) {
        this.tokenFragment = tokenFragment;
    }

    public CharFragment getCharFragment() {
        return charFragment;
    }

    public void setCharFragment(CharFragment charFragment) {
        this.charFragment = charFragment;
    }

    public String getBabelSynsetID() {
        return babelSynsetID;
    }

    public void setBabelSynsetID(String babelSynsetID) {
        this.babelSynsetID = babelSynsetID;
    }

    public String getBabelNetURL() {
        return BabelNetURL;
    }

    public void setBabelNetURL(String babelNetURL) {
        BabelNetURL = babelNetURL;
    }

    public String getScore() {
        return score;
    }

    public void setScore(String score) {
        this.score = score;
    }

    public String getCoherenceScore() {
        return coherenceScore;
    }

    public void setCoherenceScore(String coherenceScore) {
        this.coherenceScore = coherenceScore;
    }

    public String getGlobalScore() {
        return globalScore;
    }

    public void setGlobalScore(String globalScore) {
        this.globalScore = globalScore;
    }

    public String getSource() {
        return source;
    }

    public void setSource(String source) {
        this.source = source;
    }

    public String getDBpediaURL() {
        return DBpediaURL;
    }

    public void setDBpediaURL(String DBpediaURL) {
        this.DBpediaURL = DBpediaURL;
    }

    public static class TokenFragment {
        private String start;
        private String end;

        public String getStart() {
            return start;
        }

        public void setStart(String start) {
            this.start = start;
        }

        public String getEnd() {
            return end;
        }

        public void setEnd(String end) {
            this.end = end;
        }
    }

    public static class CharFragment {
        private String start;
        private String end;

        public String getStart() {
            return start;
        }

        public void setStart(String start) {
            this.start = start;
        }

        public String getEnd() {
            return end;
        }

        public void setEnd(String end) {
            this.end = end;
        }
    }
}
