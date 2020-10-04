package com.optimization.objects;

import org.apache.commons.math3.linear.OpenMapRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 * A linear function b.x + c; The gradient is b, and the Hessian is |0| (all zeros).
 */
public class LinearFunction extends ConvexFunction {
    private final RealVector b;
    private final double c;
    private final int n;

    /**
     * Construct a linear function b.x + c
     *
     * @param b a weight vector
     * @param c a constant
     */
    public LinearFunction(RealVector b, double c) {
        int d = b.getDimension();
        if (d < 1) throw new IllegalArgumentException("Dimension must be nonzero");
        this.b = b;
        this.c = c;
        this.n = d;
    }

    @Override
    public int dimensions() { return n; }

    @Override
    public double value(final RealVector x) {
        return c + b.dotProduct(x);
    }

    @Override
    public RealVector gradient(final RealVector x) {
        return b.copy();
    }

    @Override
    public RealMatrix hessian(final RealVector x) {
        // the Hessian is just zero for a linear function
        return new OpenMapRealMatrix(n, n);
    }

    @Override
    public String toString() {
        return String.format("LinearFunction(%g, %s)", c, b.toString());
    }
}
