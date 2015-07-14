import weka.core.Instance;

/**
 * Created by lyz on 7/13/15.
 */
public class Utils {
    public static double[] getFeatures(Instance ins) {
        double[] features = new double[ins.numAttributes() - 1];
        int k = 0;
        for (int i = 0; i < ins.numAttributes(); i++) {
            if (i != ins.classIndex()) {
                features[k++] = ins.value(i);
            }
        }
        return features;
    }

    public static void normalizeVectorInPlace(double[] array) {
        double norm = Utils.norm(array);
        for (int i = 0; i < array.length; i++) {
            array[i] /= norm;
        }
    }

    public static double squareSum(double[] array) {
        double sum = 0;
        for (double v : array) {
            sum += v * v;
        }
        return sum;
    }

    public static double norm(double[] array) {

        return Math.sqrt(squareSum(array));
    }

    public static double[] vectorPlus(double[] array1, double[] array2) {
        if (array1.length != array2.length) {
            return null;
        }
        double[] sum = new double[array1.length];
        for (int i = 0; i < array1.length; i++) {
            sum[i] = array1[i] + array2[i];
        }
        return sum;
    }

    public static double[] vectorMinus(double[] array1, double[] array2) {
        if (array1.length != array2.length) {
            return null;
        }
        double[] differ = new double[array1.length];
        for (int i = 0; i < array1.length; i++) {
            differ[i] = array1[i] - array2[i];
        }
        return differ;
    }

    public static double[] vectorTimesScalar(double[] array1, double number) {

        double[] product = new double[array1.length];
        for (int i = 0; i < array1.length; i++) {
            product[i] = array1[i] * number;
        }
        return product;
    }

    public static double dotProduct(double[] array1, double[] array2) {
        if (array1.length != array2.length) {
            return 0;
        }
        double product = 0;
        for (int i = 0; i < array1.length; i++) {
            product += array1[i] * array2[i];
        }
        return product;
    }

    public static int max_index(double[] array) {
        int index = 0;
        double max = array[0];
        for (int i = 1; i < array.length; i++) {
            if (array[i] > max) {
                max = array[i];
                index = i;
            }
        }
        return index;
    }
}
