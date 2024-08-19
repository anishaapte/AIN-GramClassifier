import java.io.FileReader;
import java.math.RoundingMode;
import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.Random;
/**
 * Project A1
 * @author anishaapte
 *
 */
public class NGramClassifier {

    static int[] unigramCounts;
    double[] unigramProb;
    String[] unigramProbStr;

    static int[][] bigramCounts;
    double[][] bigramNoSmooth;
    String[][] noSmoothFinalStr;
    double[][] bigramSmooth;
    String[][] smoothFinalStr;

    int[][][] trigramCounts;
    double[][][] trigramProb;
    String[][][] trigramProbStr;

    String[] genSentence;

    String fakeScriptStr;
    int[] fakeScriptCount;
    double[] fakeScriptProb;

    public static final double PROB_FAKE = 0.12;
    public static final double PROB_REAL = 0.88;

    double[] postProb;
    int[] docPredictions;

    public static void main(String[] args) {
        
        NGramClassifier ng = new NGramClassifier();

        String newScript = ng.processText("/Users/anishaapte/Desktop/AI540/joker_script.txt");
        ng.fakeScript("/Users/anishaapte/Desktop/AI540/fake.txt");
        ng.naiveBayes();
        ng.createUnigram(newScript);
        ng.createBigram(newScript, unigramCounts);
        ng.createTrigram(newScript, bigramCounts);
        ng.genSentences();
        ng.calcPosterior();
        ng.predictAllSentences();
        ng.printAnswers();

    }
    
    public NGramClassifier() {
        
    }

    public  String processText(String filename) {
        try {
            StringBuffer finalScript = new StringBuffer();
            FileReader f = new FileReader(filename);
            int curr = f.read();
            boolean foundSpace = false;
            while (curr != -1) {
                char currChar = (char) curr;
                // only printing letters and spaces
                if ((currChar >= 'a' && currChar <= 'z') || (currChar >= 'A' && currChar <= 'Z') || (currChar == ' ')) {

                    // converts uppercase to lowercase
                    if (currChar >= 'A' && currChar <= 'Z') {
                        currChar = Character.toLowerCase(currChar);
                    }
                    // removes all consecutive spaces, only a single space remains
                    if (currChar == ' ') {
                        if (foundSpace == false) {
                            foundSpace = true;
                        } else {
                            curr = f.read();
                            continue;
                        }
                    } else {
                        foundSpace = false;
                    }
                    finalScript.append(currChar);
                }
                curr = f.read();
            }
            return finalScript.toString();

        } catch (Throwable e) {
            e.printStackTrace();
        }

        return null;

    }

    public void createUnigram(String s) {

        unigramCounts = new int[27];
        for (int i = 0; i < s.length(); i++) {
            char curr = s.charAt(i);
            if (curr == ' ') {
                unigramCounts[0]++;
            } else {
                int currInt = (int) curr;
                unigramCounts[currInt - 96]++;
            }
        }

        
        unigramProb = new double[27];
        double totalProb = 0.0;

        for (int i = 0; i < unigramCounts.length; i++) {
            unigramProb[i] = ((double) unigramCounts[i]) / ((double) s.length());
            unigramProb[i] = ((double) (Math.round((unigramProb[i] * 10000)))) / 10000.0;
            if (unigramProb[i] < 0.0001) {
                unigramProb[i] = 0.0001;
            }
            totalProb += unigramProb[i];
        }

        totalProb = ((double) (Math.round((totalProb * 10000)))) / 10000.0;

        if (totalProb != 1.0) {
            unigramProb[0] -= totalProb - 1.0;
            unigramProb[0] = ((double) (Math.round((unigramProb[0] * 10000)))) / 10000.0;
        }

        unigramProbStr = new String[27];
        for (int i = 0; i < unigramProb.length; i++) {
            unigramProbStr[i] = String.format("%.4f", unigramProb[i]);
        }

        
    }

    public void createBigram(String s, int[] uniCount) {
        bigramCounts = new int[27][27];
        bigramNoSmooth = new double[27][27];
        noSmoothFinalStr = new String[27][27];
        smoothFinalStr = new String[27][27];
        bigramSmooth = new double[27][27];

        for (int i = 0; i < s.length() - 1; i++) {
            int rowIndex = 0;
            int colIndex = 0;
            if (s.charAt(i) != ' ') {
                rowIndex = ((int) (s.charAt(i))) - 96;
            }
            if (s.charAt(i + 1) != ' ') {
                colIndex = ((int) (s.charAt(i + 1))) - 96;
            }
            bigramCounts[rowIndex][colIndex]++;

        }
        for (int i = 0; i < bigramCounts.length; i++) {
            for (int j = 0; j < bigramCounts[i].length; j++) {
                bigramNoSmooth[i][j] = ((double) bigramCounts[i][j]) / ((double) uniCount[i]);
                bigramNoSmooth[i][j] = ((double) (Math.round(bigramNoSmooth[i][j] * 10000.0))) / 10000.0;

                noSmoothFinalStr[i][j] = String.format("%.4f", bigramNoSmooth[i][j]);
            }
        }

        for (int i = 0; i < bigramCounts.length; i++) {
            for (int j = 0; j < bigramCounts[i].length; j++) {
                bigramSmooth[i][j] = (((double) (bigramCounts[i][j]) + 1)) / ((double) (uniCount[i] + 27));
                bigramSmooth[i][j] = ((double) (Math.round(bigramSmooth[i][j] * 10000.0))) / 10000.0;
                if (bigramSmooth[i][j] < 0.0001) {
                    bigramSmooth[i][j] = 0.0001;
                }
                smoothFinalStr[i][j] = String.format("%.4f", bigramSmooth[i][j]);
            }
        }
    }

    public void createTrigram(String script, int[][] biCount) {
        trigramCounts = new int[27][27][27];
        trigramProb = new double[27][27][27];
        trigramProbStr = new String[27][27][27];
        for (int i = 0; i < script.length() - 2; i++) {
            int curr = 0;
            int firstNext = 0;
            int secondNext = 0;
            if (script.charAt(i) != ' ') {
                curr = ((int) (script.charAt(i))) - 96;
            }
            if (script.charAt(i + 1) != ' ') {
                firstNext = ((int) (script.charAt(i + 1))) - 96;
            }
            if (script.charAt(i + 2) != ' ') {
                secondNext = ((int) (script.charAt(i + 2))) - 96;
            }
            trigramCounts[curr][firstNext][secondNext]++;

        }
        for (int i = 0; i < trigramCounts.length; i++) {
            for (int j = 0; j < trigramCounts[i].length; j++) {
                for (int k = 0; k < trigramCounts[i][j].length; k++) {
                    if (((double) biCount[i][j]) == 0.0) {
                        trigramProb[i][j][k] = 0.0;
                    } else {
                        trigramProb[i][j][k] = (((double) trigramCounts[i][j][k] + 1))
                                / (((double) biCount[i][j] + 27));
                        trigramProb[i][j][k] = ((double) (Math.round(trigramProb[i][j][k] * 10000.0))) / 10000.0;
                        trigramProbStr[i][j][k] = String.format("%.4f", trigramProb[i][j][k]);
                    }

                }

            }

        } // end for

    }

    public void genSentences() {
        genSentence = new String[26];
        char first = 'a';
        for (int i = 0; i < 26; i++) {
            genSentence[i] = "" + first;
            genSentence[i] += getNextCharFromBi(first);
            genSentence[i] = fill998Chars(genSentence[i]);
            first++;
        }
    }

    private char getNextCharFromBi(char prev) {
        Random num = new Random();
        double rand = num.nextDouble();
        int rowIndex = 0;
        if (prev != ' ') {
            rowIndex = ((int) prev) - 96;
        }
        double[] cdf = computeCDF(bigramSmooth[rowIndex]);

        for (int i = 0; i < cdf.length; i++) {
            if (cdf[i] > rand) {
                if (i == 0) {
                    return ' ';
                } else {
                    return ((char) (i + 96));
                }

            }
        }

        return 'z';
    }

    private String fill998Chars(String s) {
        String retVal = s;
        for (int i = 2; i < 1000; i++) {
            char last = retVal.charAt(i - 1);
            char secLast = retVal.charAt(i - 2);
            int lastInd = 0;
            int secLastInd = 0;

            if (last != ' ') {
                lastInd = ((int) last) - 96;
            }
            if (secLast != ' ') {
                secLastInd = ((int) secLast) - 96;
            }

            if (bigramSmooth[secLastInd][lastInd] == 0.0) {
                retVal += getNextCharFromBi(last);
            } else {
                retVal += getNextFromTri(lastInd, secLastInd, last, secLast);
            }

        }
        return retVal;

    }

    private char getNextFromTri(int lastInd, int secLastInd, char last, char secLast) {
        double[] cdf = computeCDF(trigramProb[secLastInd][lastInd]);
        Random r = new Random();
        double rand = r.nextDouble();

        for (int i = 0; i < cdf.length; i++) {
            if (cdf[i] > rand) {
                if (i == 0) {
                    return ' ';
                } else {
                    return ((char) (i + 96));
                }

            }
        }
        return 'z';

    }
    /**
     * 
     * Credit to A1 FAQ : A1.java
     * 
     * @param origProb
     * @return
     */
    private double[] computeCDF(double[] origProb) {
        double sum = 0.0;
        double[] cdf = new double[27];
        for (int i = 0; i < 27; i++) {
            sum += origProb[i];
            cdf[i] = sum;
        }
        return cdf;
    }

    public void fakeScript(String filename) {
        try {
            FileReader f = new FileReader(filename);
            StringBuffer fakeFile = new StringBuffer();
            int curr = f.read();
            while (curr != -1) {
                char currChar = (char) curr;
                fakeFile.append(currChar);
                curr = f.read();
            }
            fakeScriptStr = fakeFile.toString();
        } catch (Throwable e) {
            e.printStackTrace();
        }

    }

    public void naiveBayes() {
        fakeScriptCount = new int[27];
        fakeScriptProb = new double[27];
        for (int i = 0; i < fakeScriptStr.length(); i++) {
            char curr = fakeScriptStr.charAt(i);
            if (curr == ' ') {
                fakeScriptCount[0]++;
            } else {
                int currInt = (int) curr;
                if ((currInt - 96) >= 0 && (currInt - 96) <= 26) {
                    fakeScriptCount[currInt - 96]++;
                }
            }
        }

        for (int i = 0; i < fakeScriptCount.length; i++) {
            fakeScriptProb[i] = fakeScriptCount[i] / 10000.0;
            fakeScriptProb[i] = ((double) (Math.round((fakeScriptProb[i] * 10000)))) / 10000.0;

        }


    }

    public void calcPosterior() {
        postProb = new double[27];
        for (int i = 0; i < postProb.length; i++) {
            postProb[i] = (PROB_FAKE * fakeScriptProb[i])
                    / ((PROB_FAKE * fakeScriptProb[i] + (PROB_REAL * unigramProb[i])));
            postProb[i] = ((double) (Math.round((postProb[i] * 10000)))) / 10000.0;
        }

    }

    public int predictDocu(String s) {
        double realSum = 0.0;
        double fakeSum = 0.0;

        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == ' ') {
                fakeSum += Math.log(fakeScriptProb[0]);
                realSum += Math.log(unigramProb[0]);
            } else {
                int index = ((int) s.charAt(i)) - 96;
                fakeSum += Math.log(fakeScriptProb[index]);
                realSum += Math.log(unigramProb[index]);
            }
        }

        if (fakeSum > realSum) {
            return 1;
        } else {
            return 0;
        }

    }

    public void predictAllSentences() {
        docPredictions = new int[26];
        for (int i = 0; i < docPredictions.length; i++) {
            docPredictions[i] = predictDocu(genSentence[i]);
        }
    }

    public void printArr(String s) {
        System.out.println(s.substring(1, s.length() - 1));
    }

    public void printAnswers() {
        System.out.println("================================================");
        System.out.println("Q2.1 ----");
        printArr(Arrays.toString(unigramProbStr));

        System.out.println("Q3.1 ----");
        for (int i = 0; i < noSmoothFinalStr.length; i++) {
            printArr(Arrays.toString(noSmoothFinalStr[i]));
        }

        System.out.println("Q4.1 ----");
        for (int i = 0; i < smoothFinalStr.length; i++) {
            printArr(Arrays.toString(smoothFinalStr[i]));
        }

        System.out.println("Q5.1 ----");
        for (int i = 0; i < genSentence.length; i++) {
            System.out.println(genSentence[i]);
        }

        System.out.println("Q7.1 ----");
        printArr(Arrays.toString(fakeScriptProb));

        System.out.println("Q8.1 ----");
        printArr(Arrays.toString(postProb));

        System.out.println("Q9.1 ----");
        printArr(Arrays.toString(docPredictions));

    }
}
