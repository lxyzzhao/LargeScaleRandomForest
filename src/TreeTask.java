import weka.core.Instances;

/**
 * Created by lyz on 7/14/15.
 */
public class TreeTask {
    private RandomTree.Tree parent;
    private Instances dataSet;
    private int child_index = 0;
    private int depth=0;

    public int getDepth() {
        return depth;
    }

    public void setDepth(int depth) {
        this.depth = depth;
    }

    public RandomTree.Tree getParent() {
        return parent;
    }

    public void setParent(RandomTree.Tree parent) {
        this.parent = parent;
    }

    public Instances getDataSet() {
        return dataSet;
    }

    public void setDataSet(Instances dataSet) {
        this.dataSet = dataSet;
    }

    public int getChild_index() {
        return child_index;
    }

    public void setChild_index(int child_index) {
        this.child_index = child_index;
    }


}
