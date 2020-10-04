package com.optimization.objects;

import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.optim.OptimizationData;

public class LinearInequalityConstraint implements OptimizationData {
    /** The corresponding set of individual linear constraint functions */
    public final LinearFunction[] lcf;

    /**
     * Construct a set of linear inequality constraints from Ax &lt; B
     * @param A A matrix linear coefficient vectors
     * @param b A vector of constants
     */
    public LinearInequalityConstraint(final RealMatrix A, final RealVector b) {
        int k = A.getRowDimension();
        if (b.getDimension() != k) {
            throw new DimensionMismatchException(b.getDimension(), k);
        }
        this.lcf = new LinearFunction[k];
        for (int j = 0; j < k; ++j) {
            lcf[j] = new LinearFunction(A.getRowVector(j), -b.getEntry(j));
        }
    }
}