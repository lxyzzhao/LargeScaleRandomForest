import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;

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

        Normalize filter = new Normalize();
        try {
            filter.setInputFormat(train);
          //  train = Filter.useFilter(train, filter);
        } catch (Exception e) {
            e.printStackTrace();
        }

        System.out.println("Read train data finished");
        Instances newTrain = train.testCV(40, 1);

        System.out.println("Begin Random Forest");
        RandomForest rf = new RandomForest();
        try {
            rf.setAccuracy_e(0.01);
            rf.setMax_depth(50);//Integer.MAX_VALUE);
            rf.setMin_Instance(10);
            rf.setMax_trees(100);
            rf.buildClassifier(newTrain);
        } catch (Exception e) {
            e.printStackTrace();
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
        try {
           // test = Filter.useFilter(test, filter);
        } catch (Exception e) {
            e.printStackTrace();
        }
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
