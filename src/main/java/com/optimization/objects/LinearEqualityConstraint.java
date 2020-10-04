package com.optimization.objects;

import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.optim.OptimizationData;

/**
 * Represents a set of linear equality constraints given as Ax = b.
 */
public class LinearEqualityConstraint implements OptimizationData {
    public final RealMatrix A;
    public final RealVector b;

    /**
     * Construct a set of linear equality constraints Ax = b.
     * Represents equations A[i].x = b[i], for each row of A.
     * @param A the matrix of linear weights
     * @param b the vector of constants
     */
    public LinearEqualityConstraint(final RealMatrix A, final RealVector b) {
        int k = A.getRowDimension();
        if (b.getDimension() != k)
            throw new DimensionMismatchException(b.getDimension(), k);
        this.A = A;
        this.b = b;
    }

    /**
     * Construct a set of linear equality constraints Ax = b.
     * Represents equations A[i].x = b[i], for each row of A.
     * @param A the matrix of linear weights
     * @param b the vector of constants
     */
    public LinearEqualityConstraint(final double[][] A, final double[] b) {
        this(new Array2DRowRealMatrix(A), new ArrayRealVector(b));
    }
}
