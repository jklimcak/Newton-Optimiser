package com.optimization.objects;

import com.optimization.util.SmoothMaxFunction;
import org.apache.commons.math3.analysis.MultivariateFunction;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.optim.InitialGuess;
import org.apache.commons.math3.optim.OptimizationData;
import org.apache.commons.math3.optim.PointValuePair;
import org.apache.commons.math3.optim.nonlinear.scalar.MultivariateOptimizer;
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunction;

import java.util.ArrayList;

public abstract class ConvexOptimizer extends MultivariateOptimizer {
    protected ConvexFunction convexObjective;

    protected ConvexOptimizer() {
        super(null);
    }

    @Override
    protected void parseOptimizationData(OptimizationData... optData) {
        super.parseOptimizationData(optData);
        for (OptimizationData data: optData) {
            if (data instanceof ObjectiveFunction) {
                MultivariateFunction f = ((ObjectiveFunction)data).getObjectiveFunction();
                if (f instanceof ConvexFunction) {
                    convexObjective = (ConvexFunction)f;
                } else {
                    throw new IllegalArgumentException("ConvexFunction objective required");
                }
            }
        }
        if (convexObjective == null)
            throw new IllegalStateException("Expected a ConvexFunction argument");
    }

    @Override
    public PointValuePair optimize(OptimizationData... optData) {
        return super.optimize(optData);
    }

    public static PointValuePair feasiblePoint(OptimizationData... optData) {
        double epsilon = 1e-9;
        RealVector initialGuess = null;
        ArrayList<ConvexFunction> ineqConstraints = new ArrayList<ConvexFunction>();
        final ConvexFunction[] fType = new ConvexFunction[0];
        final OptimizationData[] odType = new OptimizationData[0];
        final ArrayList<OptimizationData> solverArgs = new ArrayList<OptimizationData>();

        for (OptimizationData data: optData) {
            if (canPassFromMain(data)) {
                solverArgs.add(data);
            }
            if (data instanceof LinearInequalityConstraint) {
                for (ConvexFunction f: ((LinearInequalityConstraint)data).lcf)
                    ineqConstraints.add(f);
            }
        }

        // check to see if any linear equality constraints are in effect
        boolean hasLinearInequalityConstraint = false;
        for (OptimizationData data: solverArgs.toArray(odType))
            if (data instanceof LinearEqualityConstraint) {
                hasLinearInequalityConstraint = true;
            }
        final int n = ineqConstraints.get(0).dimensions();
        initialGuess = new ArrayRealVector(n, 0.0);
        final ConvexFunction[] fk = ineqConstraints.toArray(fType);
        final double minNBallFactor = Math.log(1e-3);
        RealVector x = initialGuess;
        double s = fkMax(x.toArray(), fk);

        double alpha = 1.0;
        while (true) {
            double radius = Math.max(1.0, 2.0 * s);
            ConvexFunction nbc = QuadraticFunction.nBallConstraintFunction(x, radius, 1.0 / (radius * radius));
            ArrayList<ConvexFunction> augConstraints = (ArrayList<ConvexFunction>)(ineqConstraints.clone());
            augConstraints.add(nbc);

            double v0 = nbc.value(x);
            if (v0 < (s + minNBallFactor)) {
                alpha = minNBallFactor / (v0 - s);
            }

            ArrayList<OptimizationData> args = (ArrayList<OptimizationData>)solverArgs.clone();
            args.add(new InitialGuess(x.toArray()));
            args.add(new ObjectiveFunction(new SmoothMaxFunction(alpha, augConstraints.toArray(fType))));
            PointValuePair spvp = (new NewtonOptimizer()).optimize(args.toArray(odType));
            RealVector xprv = x;
            x = new ArrayRealVector(spvp.getFirst());
            s = fkMax(spvp.getFirst(), fk);
            if (s < 0.0) {
                break;
            }
            RealVector xdelta = x.subtract(xprv);
            if (xdelta.dotProduct(xdelta) < epsilon) {
                break;
            }

            // increase alpha as we converge, so that smooth-max more closely approximates true max
            // see: http://erikerlandson.github.io/blog/2019/01/02/the-smooth-max-minimum-incident-of-december-2018/
            alpha *= 10.0;
        }

        return new PointValuePair(x.toArray(), s);
    }

    private static boolean canPassFromMain(OptimizationData data) {
        if (data instanceof InitialGuess) return false;
        if (data instanceof ObjectiveFunction) return false;
        return true;
    }

    private static double fkMax(double[] x, ConvexFunction[] fk) {
        double s = Double.NEGATIVE_INFINITY;
        for (ConvexFunction f: fk) {
            double y = f.value(x);
            if (s < y) s = y;
        }
        return s;
    }
}
