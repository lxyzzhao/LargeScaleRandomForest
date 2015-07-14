import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Enumeration;
import java.util.Queue;

/**
 * Created by lyz on 7/13/15.
 */
public class RandomTree {

    private int min_Instance = 10;

    private double accuracy_e = 0.01;

    private double max_depth = Integer.MAX_VALUE;

    private Tree root = null;

    public void setMin_Instance(int min_Instance) {
        this.min_Instance = min_Instance;
    }

    public void setAccuracy_e(double accuracy_e) {
        this.accuracy_e = accuracy_e;
    }

    public void setMax_depth(double max_depth) {
        this.max_depth = max_depth;
    }

    public void buildClassifier(Instances instances) {
        root = new Tree();
        root.min_Instance = min_Instance;
        root.accuracy_e = accuracy_e;
        root.max_depth = max_depth;

        root.makeTree(instances);
    }

    public double[] distribution(Instance instance) {
        return root.distribution(instance);
    }

    class Tree {
        private int min_Instance = 10;
        private double accuracy_e = 0.01;
        private double max_depth = Integer.MAX_VALUE;

        private Tree[] subTrees;
        private LinearModel model = null;
        double[] distribution;

        public void makeTree(Instances instances) {
            makeTree(instances, 0);
        }

        private void makeTree(Instances instances, int depth) {
          //  System.out.println("Current Depth: "+depth);

            distribution = distributionFor(instances);
            if (instances.numInstances() <= min_Instance || depth > max_depth) {
                return;
            }
            for (int i = 0; i < instances.numClasses(); i++) {
                if (distribution[i] == 1) {
                    return;
                }
            }

            model = new LinearModel();
            model.setAccuracy_e(accuracy_e);
            model.buildClassifier(instances);

            Instances[] subsets = splitData(instances);

            subTrees = new Tree[2];
            for (int i = 0; i < 2; i++) {
                subTrees[i] = new Tree();
                subTrees[i].min_Instance = min_Instance;
                subTrees[i].accuracy_e = accuracy_e;
                subTrees[i].max_depth = max_depth;
                subTrees[i].makeTree(subsets[i], depth + 1);
            }
            distribution = null;
        }

        private Instances[] splitData(Instances data) {
            Instances[] subsets = new Instances[2];
            for (int i = 0; i < 2; i++) {
                subsets[i] = new Instances(data, data.numInstances());
            }

            Enumeration enumeration = data.enumerateInstances();
            while (enumeration.hasMoreElements()) {
                Instance ins = (Instance) enumeration.nextElement();
                int predictedValue = model.predict(ins);
                subsets[predictedValue >= 0 ? 0 : 1].add(ins);
            }
            return subsets;
        }

        private double[] distributionFor(Instances instances) {
            double[] dist = new double[instances.numClasses()];
            Enumeration enumeration = instances.enumerateInstances();
            while (enumeration.hasMoreElements()) {
                Instance ins = (Instance) enumeration.nextElement();
                dist[((int) ins.classValue())]++;
            }
            Utils.normalizeVectorInPlace(dist);
            return dist;
        }

        public double[] distribution(Instance instance) {
            if (subTrees == null) {
                return distribution;
            } else {
                int predicted_value = model.predict(instance);
                return subTrees[predicted_value >= 0 ? 0 : 1].distribution(instance);
            }
        }
    }
}
