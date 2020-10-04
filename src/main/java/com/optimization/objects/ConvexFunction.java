package com.optimization.objects;

import org.apache.commons.math3.analysis.MultivariateFunction;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 * A [apache.commons.math3.analysis].MultivariateFunction with a defined gradient and Hessian
 *
 *  - Objective function f has to be twice differentiable
 *  - Its Hessian, or second derivative of f exists at each point in the dom f
 *
 *   -------------------------------------------------------------------------------------------------------------------
 *   | Hessian Matrix                                                                                                  |
 *   |   - A square (n x n) matrix of second-order partial derivatives of a scalar-valued function f(x1, x2, ..., xn)  |
 *   -------------------------------------------------------------------------------------------------------------------
 */
public abstract class ConvexFunction implements MultivariateFunction {
    /**
     * @return the expected dimension of the function's domain
     */
    public abstract int dimensions();

    /**
     * @param x - point at which to evaluate function.
     * @return the value of this function at (x)
     */
    public abstract double value(final RealVector x);

    /**
     * @param x - point at which to evaluate gradient.
     * @return the gradient of this function at (x)
     */
    public abstract RealVector gradient(final RealVector x);

    /**
     * @param x - point at which to evaluate Hessian.
     * @return the Hessian of this function at (x)
     */
    public abstract RealMatrix hessian(final RealVector x);

    /**
     * @param x a point to evaluate this function at.
     * @return the value of this function at (x)
     */
    @Override
    public double value(final double[] x) {
        return value(new ArrayRealVector(x, false));
    }
}
