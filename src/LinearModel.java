import weka.core.Instance;
import weka.core.Instances;

import java.util.*;

/**
 * Created by lyz on 7/13/15.
 */
public class LinearModel {

    private double[] w;
    private double b;
    private double accuracy_e = 0.01;
    private int positive_class;

    public void setAccuracy_e(double accuracy_e) {
        this.accuracy_e = accuracy_e;
    }

    public void buildClassifier(Instances data) {

        Instances dataSet = new Instances(data);

        Random random = new Random();
        dataSet.randomize(random);

        Set<Integer> set = new HashSet<>();
        Enumeration enumeration = dataSet.enumerateInstances();
        while (enumeration.hasMoreElements()) {
            Instance ins = (Instance) enumeration.nextElement();
            set.add((int) ins.classValue());
        }
        List<Integer> existed_class = new ArrayList<>(set);
        Collections.sort(existed_class);
        positive_class = existed_class.get(random.nextInt(existed_class.size()));
        set.clear();
        existed_class.clear();

        int max_iteration = (int) (1 / Math.pow(accuracy_e, 2));

        Instance pos_instance = null;
        Instance neg_instance = null;

        enumeration = dataSet.enumerateInstances();
        while (enumeration.hasMoreElements()) {
            pos_instance = (Instance) enumeration.nextElement();
            if (pos_instance.classValue() == positive_class) {
                break;
            }
        }

        enumeration = dataSet.enumerateInstances();
        while (enumeration.hasMoreElements()) {
            neg_instance = (Instance) enumeration.nextElement();
            if (neg_instance.classValue() != positive_class) {
                break;
            }
        }

        w = new double[dataSet.numAttributes() - 1];
        double[] x_pos = Utils.getFeatures(pos_instance);
        double[] x_neg = Utils.getFeatures(neg_instance);


        w = Utils.vectorMinus(x_pos, Utils.vectorTimesScalar(x_neg, 1.0 / Utils.norm(Utils.vectorMinus(x_pos, x_neg))));
        Utils.normalizeVectorInPlace(w);
        b = (Utils.dotProduct(w, x_pos) + Utils.dotProduct(w, x_neg)) / 2;

        int iteration = 0;
        int D = dataSet.numInstances();
        double[] lastW = new double[w.length];

        while (iteration < max_iteration) {
            iteration++;

            if (iteration % D == 0) {
                if (Utils.norm(Utils.vectorMinus(w, lastW)) <= 0.001) {
                    break;
                } else {
                    lastW = Arrays.copyOf(w, w.length);
                }
            }

            Instance ins = dataSet.instance(iteration % D);
            int l_x = ins.classValue() == positive_class ? +1 : -1;
            double[] x = Utils.getFeatures(ins);
            if ((Utils.dotProduct(w, x) + b) * l_x < 0) {
                double coefficient = 0.1 / (0.1 + iteration * Math.pow(accuracy_e, 2)) * l_x;
                Utils.vectorTimesScalarInPlace(x, coefficient);
                Utils.vectorPlusInPlace(w, x);
                b += coefficient;
            }
        }
    }

    public int predict(Instance ins) {
        double[] x = Utils.getFeatures(ins);
        double predict_value = Utils.dotProduct(w, x) + b;
        return predict_value > 0 ? positive_class : -1;
    }
}
