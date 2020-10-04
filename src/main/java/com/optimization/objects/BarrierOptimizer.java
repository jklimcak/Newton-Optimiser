package com.optimization.objects;

import com.optimization.util.LogBarrierFunction;
import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.optim.InitialGuess;
import org.apache.commons.math3.optim.OptimizationData;
import org.apache.commons.math3.optim.PointValuePair;
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunction;

import java.util.ArrayList;

public class BarrierOptimizer extends ConvexOptimizer {
    private ArrayList<ConvexFunction> constraintFunctions = new ArrayList<ConvexFunction>();
    private RealVector xStart;

    private double epsilon = 1e-9; // ConvergenceEpsilon.CONVERGENCE_EPSILON_DEFAULT;

    // * The mu parameter for the Barrier Method used by {@link BarrierOptimizer}.
    // * This is the scaling factor for the objective function multiplier (t),
    // * as described in (Algorithm 11.1) from <p>
    // * Convex Optimization, Boyd and Vandenberghe, Cambridge University Press, 2008.
    // * The value of mu, the scaling factor for the objective function multiplier */
    private double mu = 15.0; // BarrierMu.BARRIER_MU_DEFAULT;
    // * The initial value for t, the objective function multiplier */
    private double t0 = 1.0; // BarrierMu.BARRIER_T0_DEFAULT;


    private OptimizationData[] odType = new OptimizationData[0];
    private ArrayList<OptimizationData> newtonArgs = new ArrayList<OptimizationData>();

    public BarrierOptimizer() {
        super();
    }

    @Override
    public PointValuePair optimize(OptimizationData... optData) {
        return super.optimize(optData);
    }

    private boolean canPassFromMain(OptimizationData data) {
        if (data instanceof ObjectiveFunction) return false;
        if (data instanceof InitialGuess) return false;
        return true;
    }

    @Override
    protected void parseOptimizationData(OptimizationData... optData) {
        super.parseOptimizationData(optData);
        for (OptimizationData data : optData) {
            if (canPassFromMain(data)) {
                newtonArgs.add(data);
            }
            if (data instanceof LinearInequalityConstraint) {
                for (ConvexFunction f : ((LinearInequalityConstraint) data).lcf) {
                    constraintFunctions.add(f);
                }
            }
        }
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
        double m = (double)((constraintFunctions != null) ? constraintFunctions.size() : 0);

        RealVector x = xStart;
        for (double t = t0; (t * epsilon) <= m ; t *= mu) {
            ConvexFunction bf = new LogBarrierFunction(t, convexObjective, constraintFunctions);
            NewtonOptimizer newton = new NewtonOptimizer();
            ArrayList<OptimizationData> args = (ArrayList<OptimizationData>)newtonArgs.clone();
            args.add(new ObjectiveFunction(bf));
            args.add(new InitialGuess(x.toArray()));
            PointValuePair pvp = newton.optimize(args.toArray(odType));

            x = new ArrayRealVector(pvp.getFirst());
        }
        return new PointValuePair(x.toArray(), convexObjective.value(x));
    }
}
