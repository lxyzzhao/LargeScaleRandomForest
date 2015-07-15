import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

/**
 * Created by lyz on 7/13/15.
 */
public class Main {
    public static void main(String[] args) {

        Instances train = null;
        try {
            train = DataSource.read("./data/train.arff");
        } catch (Exception e) {
            e.printStackTrace();
            System.err.println("Read data error");
        }
        train.setClassIndex(train.numAttributes() - 1);

        System.out.println("Read train data finished");

        System.out.println("Training Random Forest...");
        RandomForest rf = new RandomForest();
        try {
            rf.setAccuracy_e(0.01);
            rf.setMax_depth(Integer.MAX_VALUE);
            rf.setMin_Instance(500);
            rf.setMax_trees(20);
            rf.buildClassifier(train);
        } catch (Exception e) {
            e.printStackTrace();
            System.err.println("Train Random Forest error");
        }

        System.out.println("Finished Random Forest");

        Instances test = null;
        try {
            test = DataSource.read("./data/test.arff");
        } catch (Exception e) {
            e.printStackTrace();
            System.err.println("Read test data error");
        }
        test.setClassIndex(test.numAttributes() - 1);

        System.out.println("Read test data finished");

        Evaluation evaluation = null;
        try {
            evaluation = new Evaluation(train);
            evaluation.evaluateModel(rf, test);
        } catch (Exception e) {
            e.printStackTrace();
        }
        System.out.println(evaluation.toSummaryString());
    }
}
