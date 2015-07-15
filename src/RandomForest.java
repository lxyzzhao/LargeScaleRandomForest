import weka.classifiers.Classifier;
import weka.core.*;

/**
 * Created by lyz on 7/13/15.
 */
public class RandomForest extends Classifier {

    private RandomTree[] trees;
    private int max_trees = 20;
    private int min_Instance = 3;
    private double accuracy_e = 0.01;
    private int max_depth = Integer.MAX_VALUE;


    public void setMax_trees(int max_trees) {
        this.max_trees = max_trees;
    }

    public void setMin_Instance(int min_Instance) {
        this.min_Instance = min_Instance;
    }

    public void setAccuracy_e(double accuracy_e) {
        this.accuracy_e = accuracy_e;
    }

    public void setMax_depth(int max_depth) {
        this.max_depth = max_depth;
    }

    public RandomForest() {
    }


    public void buildClassifier(Instances data) throws Exception {

        trees = new RandomTree[max_trees];
        for (int i = 0; i < max_trees; i++) {
            System.out.println("Training Tree No. " + i);

            trees[i] = new RandomTree();
            trees[i].setAccuracy_e(accuracy_e);
            trees[i].setMax_depth(max_depth);
            trees[i].setMin_Instance(min_Instance);
            trees[i].buildClassifier(data);

            System.out.println("Finished Tree No. " + i + ", " + 100.0 * (i + 1) / max_trees + "% finished");
        }
    }

    public int predict(Instance instance) {
        double[] dist = new double[instance.numClasses()];

        for (int i = 0; i < max_trees; i++) {
            dist = Utils.vectorPlus(dist, trees[i].distribution(instance));
        }
        return Utils.max_index(dist);
    }

    public double classifyInstance(Instance instance) throws Exception {
        return predict(instance);
    }
}
