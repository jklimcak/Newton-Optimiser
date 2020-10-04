package com.optimization.objects;

import com.optimization.util.KKTSolution;
import com.optimization.util.KKTSolver;
import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.optim.OptimizationData;
import org.apache.commons.math3.optim.PointValuePair;

public class NewtonOptimizer extends ConvexOptimizer {
    private LinearEqualityConstraint eqConstraint;
    private KKTSolver kktSolver = new KKTSolver();
    private RealVector xStart;
    private double epsilon = 1e-9;
    private double alpha = 0.4;
    private double beta = 0.8;

    public NewtonOptimizer() {
        super();
    }

    @Override
    public PointValuePair optimize(OptimizationData... optData) {
        return super.optimize(optData);
    }

    @Override
    protected void parseOptimizationData(OptimizationData... optData) {
        super.parseOptimizationData(optData);
        for (OptimizationData data: optData) {
            if (data instanceof LinearEqualityConstraint) {
                eqConstraint = (LinearEqualityConstraint)data;
            }
        }
        // if we got here, convexObjective exists
        int n = convexObjective.dimensions();
        if (this.getStartPoint() != null) {
            xStart = new ArrayRealVector(this.getStartPoint());
            if (xStart.getDimension() != n)
                throw new DimensionMismatchException(xStart.getDimension(), n);
        } else {
            xStart = new ArrayRealVector(n, 0.0);
        }
    }

    @Override
    public PointValuePair doOptimize() {
        final int n = convexObjective.dimensions();
        if ((eqConstraint == null) || (eqConstraint.b.getDimension() < 1)) {
            // constraints Ax = b are empty
            // Algorithm 9.5: Newton's method (unconstrained)
            RealVector x = xStart;
            double v = convexObjective.value(x);
            while (true) {
                incrementIterationCount();
                RealVector grad = convexObjective.gradient(x);
                RealMatrix hess = convexObjective.hessian(x);
                KKTSolution sol = kktSolver.solve(hess, grad);
                if (sol.lambdaSquared <= (2.0 * epsilon)) {
                    break;
                }
                RealVector xDelta = sol.xDelta;
                double gdd = grad.dotProduct(xDelta);
                RealVector tx = null;
                double tv = 0.0;
                boolean foundStep = false;
                for (double t = 1.0; t >= 1e-300; t *= beta) {
                    tx = x.add(xDelta.mapMultiply(t));
                    tv = convexObjective.value(tx);
                    if (Double.isInfinite(tv)) {
                        // this is barrier convention for "outside the feasible domain",
                        // so try a smaller step
                        continue;
                    }
                    double vtt = v + (t * alpha * gdd);
                    if (vtt == v) {
                        // (t)(alpha)(gdd) is less than (v)(machine-epsilon)
                        // Further tests for improvement are going to fail
                        break;
                    }
                    if (tv <= vtt) {
                        // This step resulted in an improvement, so halt with success
                        foundStep = true;
                        break;
                    }
                }
                // If there was no forward step to make, that indicates minimum
                if (!foundStep) break;
                // Update x,v for next iteration
                double vprv = v;
                x = tx;
                v = tv;
                // if improvement becomes very small then we are converged
                if (Math.abs(1.0 - (v / vprv)) < epsilon) break;
            }
            return new PointValuePair(x.toArray(), v);
        } else {
            // constraints Ax = b are non-empty
            // Algorithm 10.2: Newton's method with equality constraints
            final RealMatrix A = eqConstraint.A;
            final RealVector b = eqConstraint.b;
            final RealMatrix AT = A.transpose();
            final int nDual = b.getDimension();
            RealVector x = xStart;
            RealVector nu = new ArrayRealVector(nDual, 0.0);
            double v = convexObjective.value(x);
            while (true) {
                incrementIterationCount();
                RealVector grad = convexObjective.gradient(x);
                double rNorm = residualNorm(x, nu, grad, A, AT, b);
                if (rNorm <= epsilon) break;
                RealMatrix hess = convexObjective.hessian(x);
                KKTSolution sol = kktSolver.solve(hess, A, AT, grad, A.operate(x).subtract(b));
                RealVector xDelta = sol.xDelta;
                RealVector nuDelta = sol.nuPlus.subtract(nu);
                RealVector tx = null;
                RealVector tnu = null;
                double tv = 0.0;
                boolean foundStep = false;
                for (double t = 1.0; t >= 1e-300; t *= beta) {
                    tx = x.add(xDelta.mapMultiply(t));
                    tv = convexObjective.value(tx);
                    if (Double.isInfinite(tv)) {
                        // this is barrier convention for "outside the feasible domain",
                        // so try a smaller step
                        continue;
                    }
                    double ftt = 1.0 - (alpha * t);
                    if (ftt == 1.0) {
                        // (t)(alpha) is less than (machine-epsilon)
                        // Further tests for improvement are going to fail
                        break;
                    }
                    tnu = nu.add(nuDelta.mapMultiply(t));
                    RealVector tgrad = convexObjective.gradient(tx);
                    double tNorm = residualNorm(tx, tnu, tgrad, A, AT, b);
                    if (tNorm <= ftt * rNorm) {
                        // This step resulted in an improvement, so halt with success
                        foundStep = true;
                        break;
                    }
                }
                // If there was no forward step to make, that indicates minimum
                if (!foundStep) break;
                // update for next iteration
                double vprv = v;
                x = tx;
                nu = tnu;
                v = tv;
                // if improvement becomes very small then we are converged
                if (Math.abs(1.0 - (v / vprv)) < epsilon) break;
            }
            return new PointValuePair(x.toArray(), v);
        }
    }

    private double residualNorm(
            RealVector x, RealVector nu, RealVector grad,
            RealMatrix A, RealMatrix AT, RealVector b) {
        RealVector r = A.operate(x).subtract(b);
        RealVector rDual = grad.add(AT.operate(nu));
        double rr = r.dotProduct(r) + rDual.dotProduct(rDual);
        return Math.sqrt(rr);
    }

}

