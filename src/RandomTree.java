import weka.core.Instance;
import weka.core.Instances;

import java.util.Enumeration;
import java.util.LinkedList;
import java.util.Queue;

/**
 * Created by lyz on 7/13/15.
 */
public class RandomTree {

    private int min_Instance = 3;

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
        double[] distribution;
        private int min_Instance = 3;
        private double accuracy_e = 0.01;
        private double max_depth = Integer.MAX_VALUE;
        private Tree[] subTrees;
        private LinearModel model = null;

        public void makeTree(Instances instances) {

            distribution = distributionFor(instances);
            if (instances.numInstances() <= min_Instance) {
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
            }

            Queue<TreeTask> queue = new LinkedList<>();

            for (int i = 0; i < subsets.length; i++) {
                TreeTask task = new TreeTask();
                task.setChild_index(i);
                task.setDataSet(subsets[i]);
                task.setParent(this);
                task.setDepth(1);
                queue.offer(task);
            }

            while (!queue.isEmpty()) {
                TreeTask task = queue.poll();
                int depth = task.getDepth();

                Instances dataSet = task.getDataSet();
                task.getParent().subTrees[task.getChild_index()] = new Tree();

                Tree root = task.getParent().subTrees[task.getChild_index()];
                root.min_Instance = min_Instance;
                root.accuracy_e = accuracy_e;
                root.max_depth = max_depth;

                root.distribution = distributionFor(dataSet);

                if (dataSet.numInstances() <= root.min_Instance || depth > root.max_depth) {
                    continue;
                }

                boolean one_class = false;
                for (int i = 0; i < dataSet.numClasses(); i++) {
                    if (root.distribution[i] == 1) {
                        one_class = true;
                        break;
                    }
                }
                if (one_class) {
                    continue;
                }

                root.model = new LinearModel();
                root.model.setAccuracy_e(root.accuracy_e);
                root.model.buildClassifier(dataSet);

                subsets = root.splitData(dataSet);
                root.subTrees = new Tree[2];

                for (int i = 0; i < subsets.length; i++) {
                    TreeTask subTask = new TreeTask();
                    subTask.setChild_index(i);
                    subTask.setDataSet(subsets[i]);
                    subTask.setParent(root);
                    subTask.setDepth(depth + 1);
                    queue.offer(subTask);
                }

                root.distribution = null;
            }
        }

        private Instances[] splitData(Instances data) {
            Instances[] subsets = new Instances[2];
            for (int i = 0; i < subsets.length; i++) {
                subsets[i] = new Instances(data, data.numInstances());
            }

            Enumeration enumeration = data.enumerateInstances();
            while (enumeration.hasMoreElements()) {
                Instance ins = (Instance) enumeration.nextElement();
                int predictedValue = model.predict(ins);
                subsets[predictedValue >= 0 ? 0 : 1].add(ins);
            }
            for (int i = 0; i < subsets.length; i++) {
                subsets[i].compactify();
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
            double[] dist = null;
            Tree root = this;
            while (root != null) {

                if (root.subTrees == null) {
                    return root.distribution;
                } else {
                    int predicted_value = root.model.predict(instance);
                    root = root.subTrees[predicted_value >= 0 ? 0 : 1];
                }
            }
            return dist;
        }
    }
}
