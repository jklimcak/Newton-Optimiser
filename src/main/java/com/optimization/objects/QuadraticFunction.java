package com.optimization.objects;

import org.apache.commons.math3.linear.*;

/**
 * Standard Quadratic Function to be optimized.
 *      - Given A, b, c                           // P, q, c in cvxopt
 *      - Implements 0.5*(x^T)A(x) + b.x + c      // 0.5*(x^T)P(x) + q.x + c in cvxopt
 *      - A(x) + b is the gradient
 *      - A is the Hessian Matrix
 */
public class QuadraticFunction extends ConvexFunction {
    private final RealMatrix A;
    private final RealVector b;
    private final double c;
    private final int n;

    /**
     * Construct quadratic function 0.5*(x^T)A(x) + b.x + c
     *
     * @param A (n x n) matrix of weights for the quadratic terms.      // 0-filled matrix in cvxopt
     * @param b Vector of weights for linear terms.                     // (variances * self.tmv) in cvxopt
     * @param c Constant
     */
    public QuadraticFunction(RealMatrix A, RealVector b, double c) {
        MatrixUtils.checkSymmetric(A, 1e-6);
        this.A = A.copy();
        this.b = b.copy();
        this.c = c;
        this.n = b.getDimension();
    }

    /**
     * Construct quadratic function 0.5*(x^T)A(x) + b.x + c using arrays
     */
    public QuadraticFunction(double[][] A, double[] b, double c) {
        this(new Array2DRowRealMatrix(A), new ArrayRealVector(b), c);
    }

    @Override
    public int dimensions() {
        return n;
    }

    /**
     * Dot product of A.x + b.x + c
     */
    @Override
    public double value(final RealVector x) {
        double val = 0.5 * (A.operate(x)).dotProduct(x);
        val += b.dotProduct(x);
        val += c;
        return val;
    }

    @Override
    public RealVector gradient(final RealVector x) {
        return (A.operate(x)).add(b);
    }

    @Override
    public RealMatrix hessian(final RealVector x) {
        return A.copy();
    }

    /**
     * Create a quadratic function that corresponds to s((x-c).(x-c) &lt; r^2).
     * That is, constrained to an n-dimensional ball of radius r, with scaling factor s.
     *
     * @param center the center of the n-ball
     * @param r a radius, &gt; 0
     * @param s a scaling constant, &gt; 0
     * @return quadratic constraint function for the n-ball constraint
     */
    public static QuadraticFunction nBallConstraintFunction(RealVector center, double r, double s) {
        int n = center.getDimension();
        double[] alls = new double[n];
        java.util.Arrays.fill(alls, s);
        RealMatrix A = new DiagonalMatrix(alls);
        RealVector b = center.mapMultiply(-s);
        double c = 0.5 * ((s * center.dotProduct(center)) - (r * r));
        return new QuadraticFunction(A, b, c);
    }
}
