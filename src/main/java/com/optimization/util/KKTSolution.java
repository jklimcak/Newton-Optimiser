package com.optimization.util;

import org.apache.commons.math3.linear.RealVector;


public class KKTSolution {
    /** The delta-x vector solved from the KKT equations */
    public final RealVector xDelta;
    /** The nu+ vector solved from the KKT equations, or null if in (alg 9.5) mode */
    public final RealVector nuPlus;
    /** The lambda-squared value from the KKT equations, or null if in (alg 10.2) mode */
    public final double lambdaSquared;

    public KKTSolution(final RealVector xd, final RealVector nup) {
        this.xDelta = xd;
        this.nuPlus = nup;
        this.lambdaSquared = 0.0;
    }

    public KKTSolution(final RealVector xd, final double lsq) {
        this.xDelta = xd;
        this.lambdaSquared = lsq;
        this.nuPlus = null;
    }
}
