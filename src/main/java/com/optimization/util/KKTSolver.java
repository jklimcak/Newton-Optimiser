package com.optimization.util;

import org.apache.commons.math3.linear.DecompositionSolver;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.SingularValueDecomposition;
import org.apache.commons.math3.optim.OptimizationData;

public class KKTSolver implements OptimizationData {
    // Algorithm 10.3
    /**
     * solve block factored matrix equation:
     * <pre>
     * | H AT | | v | = -| g |
     * | A  0 | | w |    | h |
     * </pre>
     * <p>
     * where (v, w) are primal/dual delta-x and "nu+" from algorithm 10.2 of <p>
     * Convex Optimization, Boyd and Vandenberghe, Cambridge University Press, 2008.
     * @param H Hessian matrix
     * @param A coefficient matrix of equality constraints
     * @param AT transpose of A
     * @param g gradient, corresponding to H
     * @param h constant vector block corresponding to A
     * @return solution (delta-x, nu+)
     */
    public KKTSolution solve(
            final RealMatrix H,
            final RealMatrix A, final RealMatrix AT,
            final RealVector g, final RealVector h) {
        DecompositionSolver dsH = (new SingularValueDecomposition(H)).getSolver();
        RealMatrix m1 = dsH.solve(AT);
        RealVector v1 = dsH.solve(g);
        RealMatrix S = A.multiply(m1); // -S relative to 10.3
        DecompositionSolver dsS = (new SingularValueDecomposition(S)).getSolver();
        RealVector w = dsS.solve(h.subtract(A.operate(v1))); // both sides neg, so w same
        RealVector v = dsH.solve(g.add(AT.operate(w))); // this yields -v
        v.mapMultiplyToSelf(-1.0); // correct -v to +v
        return new KKTSolution(v, w);
    }

    // step 1 of algorithm 9.5
    /**
     * Solve constraint-free system Hv = -g <p>
     * returns delta-x (aka v) and lambda-squared from Algorithm 9.5 of <p>
     * Convex Optimization, Boyd and Vandenberghe, Cambridge University Press, 2008. <p>
     * delta-nu (aka w, aka the dual) is returned as null
     * @param H Hessian matrix
     * @param g gradient, corresponding to H
     * @return solution delta-x with lambda-squared
     */
    public KKTSolution solve(final RealMatrix H, final RealVector g) {
        DecompositionSolver dsH = (new SingularValueDecomposition(H)).getSolver();
        RealVector v = dsH.solve(g);
        double lsq = g.dotProduct(v);
        v.mapMultiplyToSelf(-1.0);
        return new KKTSolution(v, lsq);
    }
}
