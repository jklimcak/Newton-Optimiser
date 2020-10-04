package com.optimization.objects;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.optim.InitialGuess;
import org.apache.commons.math3.optim.PointValuePair;
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunction;
import org.junit.Test;
import static org.junit.Assert.*;

public class ConvexFunctionTest {

    @Test
    public void test_1() {
        System.out.println(" - TEST 1: https://courses.csail.mit.edu/6.867/wiki/images/a/a7/Qp-cvxopt.pdf -");
        System.out.println(" \t + no equality conditions");

        double[] TARGET = { 0.0, 5.0 };
        QuadraticFunction q = new QuadraticFunction(
                new double[][] {
                        { 1.0, 0.0 },
                        { 0.0, 0.0 },
                },
                new double[] { 3.0, 4.0 },
                0.0);

        LinearInequalityConstraint ineqc = new LinearInequalityConstraint(
                new Array2DRowRealMatrix(new double[][] {
                        { -1.0, 0.0 },
                        { 0.0, -1.0 },
                        { -1.0, -3.0 },
                        { 2.0, 5.0 },
                        { 3.0, 4.0 }
                }),
                new ArrayRealVector(new double[] { 0.0,0.0,-15.0,100.0,80.0 }));

        PointValuePair fpvp = ConvexOptimizer.feasiblePoint(ineqc);
        assertTrue(fpvp.getSecond() < 0.0); // if not < 0, there is no feasible point

        double[] ig = fpvp.getFirst();
        BarrierOptimizer barrier = new BarrierOptimizer();
        PointValuePair pvp = barrier.optimize(
                new ObjectiveFunction(q),
                ineqc,
                new InitialGuess(ig)
        );

        double[] RESULT = pvp.getFirst();
        System.out.println("\n\t\t X1  |   " + Math.round(pvp.getFirst()[0]));
        System.out.println("\t\t X2  |   " + Math.round(pvp.getFirst()[1]));
        assertArrayEquals(TARGET, RESULT, 1e-8);
    }

    @Test
    public void test_2() {
        System.out.println("\n- TEST 2: https://courses.csail.mit.edu/6.867/wiki/images/a/a7/Qp-cvxopt.pdf -");
        System.out.println(" \t + with equality conditions");

        double[] TARGET = { 12.0, 1.0 };
        QuadraticFunction q = new QuadraticFunction(
                new double[][] {
                        { 1.0, 0.0 },
                        { 0.0, 0.0 },
                },
                new double[] { 3.0, 4.0 },
                0.0);

        LinearInequalityConstraint ineqc = new LinearInequalityConstraint(
                new Array2DRowRealMatrix(new double[][] {
                        { -1.0, 0.0 },
                        { 0.0, -1.0 },
                        { -1.0, -3.0 },
                        { 2.0, 5.0 },
                        { 3.0, 4.0 }
                }),
                new ArrayRealVector(new double[] { 0.0,0.0,-15.0,100.0,80.0 }));


        LinearEqualityConstraint eqc = new LinearEqualityConstraint(
                new double[][] { { 0.0, 1.0 } },  // constraint y = 1,
                new double[] { 1.0 });

        PointValuePair fpvp = ConvexOptimizer.feasiblePoint(ineqc, eqc);
        assertTrue(fpvp.getSecond() < 0.0); // if not < 0, there is no feasible point

        double[] ig = fpvp.getFirst();
        BarrierOptimizer barrier = new BarrierOptimizer();
        PointValuePair pvp = barrier.optimize(
                new ObjectiveFunction(q),
                ineqc,
                eqc,
                new InitialGuess(ig)
        );

        double[] RESULT = pvp.getFirst();
        System.out.println("\n\t\t X1  |  " + Math.round(pvp.getFirst()[0]));
        System.out.println("\t\t X2  |   " + Math.round(pvp.getFirst()[1]));
        assertArrayEquals(TARGET, RESULT, 1e-8);
    }

    @Test
    public void test_3() {
        System.out.println("\n- TEST 3: DivEquity -");
        System.out.println(" \t + Trade Amount: (-)$607,761");
        System.out.println(" \t + no equality conditions");
        System.out.println(" \t + USGrowth Overweight");

        double[] TARGET = {
                -114312.87038965366,
                0,
                -2122877.936040616,
                -6451589.778830449,
                -2.12766309999064E-8,
                -24005.70279964148,
                0
        };

        QuadraticFunction q = new QuadraticFunction(
                new double[][] {
                        { 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
                        { 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
                        { 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0 },
                        { 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0 },
                        { 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0 },
                        { 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0 },
                        { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0 },
                },
                new double[] {
                        228625.74077148,
                        -5961785.50912065,
                        4245755.87208081,
                        12903179.55766076,
                        -104956.80160955,
                        48011.40556201,
                        -10143315.13464012
                },
                0.0);

        LinearInequalityConstraint ineqc = new LinearInequalityConstraint(
                new Array2DRowRealMatrix(new double[][] {
                        { 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
                        { 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
                        { 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0 },
                        { 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0 },
                        { 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0 },
                        { 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0 },
                        { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 },
                }),
                new ArrayRealVector(new double[] { -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0 }));

        PointValuePair fpvp = ConvexOptimizer.feasiblePoint(ineqc);
        assertTrue(fpvp.getSecond() < 0.0);

        double[] ig = fpvp.getFirst();
        BarrierOptimizer barrier = new BarrierOptimizer();
        PointValuePair pvp = barrier.optimize(
                new ObjectiveFunction(q),
                ineqc,
                new InitialGuess(ig)
        );

        double[] RESULT = pvp.getFirst();
        System.out.println("\n\t\t CapitalValue    |  " + Math.round(pvp.getFirst()[0]));
        System.out.println("\t\t GrowthAndIncome |        " + Math.round(pvp.getFirst()[1]));
        System.out.println("\t\t MidCapGrowth    | " + Math.round(pvp.getFirst()[2]));
        System.out.println("\t\t USGrowth        | " + Math.round(pvp.getFirst()[3]));
        System.out.println("\t\t Explorer        |        " + Math.round(pvp.getFirst()[4]));
        System.out.println("\t\t WindsorII       |   " + Math.round(pvp.getFirst()[5]));
        System.out.println("\t\t Windsor         |        " + Math.round(pvp.getFirst()[6]));
        assertArrayEquals(TARGET, RESULT, 1e-8);
    }

    @Test
    public void test_4() {
        System.out.println("\n- TEST 4: DivEquity -");
        System.out.println(" \t + Trade Amount: (-)$607,761");
        System.out.println(" \t + with equality conditions");
        System.out.println(" \t + USGrowth Overweight");

        double[] TARGET = {
                0,
                0,
                0,
                -607760.9999999965,
                0,
                0,
                0
        };

        QuadraticFunction q = new QuadraticFunction(
                new double[][] {
                        { 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
                        { 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
                        { 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0 },
                        { 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0 },
                        { 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0 },
                        { 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0 },
                        { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0 },
                },
                new double[] {
                        228625.74077148,
                        -5961785.50912065,
                        4245755.87208081,
                        12903179.55766076,
                        -104956.80160955,
                        48011.40556201,
                        -10143315.13464012
                },
                0.0);

        LinearInequalityConstraint ineqc = new LinearInequalityConstraint(
                new Array2DRowRealMatrix(new double[][] {
                        { 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
                        { 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
                        { 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0 },
                        { 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0 },
                        { 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0 },
                        { 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0 },
                        { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 },
                }),
                new ArrayRealVector(new double[] { -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0 }));


        LinearEqualityConstraint eqc = new LinearEqualityConstraint(
                new double[][] {
                        {
                                -1.6453836294201175e-06,
                                -1.6453836294201175e-06,
                                -1.6453836294201175e-06,
                                -1.6453836294201175e-06,
                                -1.6453836294201175e-06,
                                -1.6453836294201175e-06,
                                -1.6453836294201175e-06
                        },
                },
                new double[] { 1.0 });

        PointValuePair fpvp = ConvexOptimizer.feasiblePoint(ineqc, eqc);
        assertTrue(fpvp.getSecond() < 0.0);
        double[] ig = fpvp.getFirst();
        BarrierOptimizer barrier = new BarrierOptimizer();
        PointValuePair pvp = barrier.optimize(
                new ObjectiveFunction(q),
                ineqc,
                eqc,
                new InitialGuess(ig)
        );

        double[] RESULT = pvp.getFirst();
        System.out.println("\n\t\t CapitalValue    |        " + Math.round(pvp.getFirst()[0]));
        System.out.println("\t\t GrowthAndIncome |        " + Math.round(pvp.getFirst()[1]));
        System.out.println("\t\t MidCapGrowth    |        " + Math.round(pvp.getFirst()[2]));
        System.out.println("\t\t USGrowth        |  " + Math.round(pvp.getFirst()[3]));
        System.out.println("\t\t Explorer        |        " + Math.round(pvp.getFirst()[4]));
        System.out.println("\t\t WindsorII       |        " + Math.round(pvp.getFirst()[5]));
        System.out.println("\t\t Windsor         |        " + Math.round(pvp.getFirst()[6]));
        assertArrayEquals(TARGET, RESULT, 1e-8);
    }

    @Test
    public void test_5() {
        System.out.println("\n- TEST 5: DivEquity -");
        System.out.println(" \t + Trade Amount: (+)$1,000,000");
        System.out.println(" \t + with equality conditions");
        System.out.println(" \t + WindsorII Underweight");

        double[] TARGET = {
                2.723858532728371E-8,
                3.196913230341922E-8,
                2.29166390438788E-8,
                3.871120995661243E-9,
                2.619137055347329E-9,
                999999.9999998814,
                3.0069055146404884E-8
        };

        QuadraticFunction q = new QuadraticFunction(
                new double[][] {
                        { 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
                        { 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
                        { 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0 },
                        { 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0 },
                        { 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0 },
                        { 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0 },
                        { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0 },
                },
                new double[] {
                        228625.74077148,
                        -5961785.50912065,
                        4245755.87208081,
                        1290317.55766076,
                        -104956.80160955,
                        -48011999.40556201,
                        -10143315.13464012
                },
                0.0);

        LinearInequalityConstraint ineqc = new LinearInequalityConstraint(
                new Array2DRowRealMatrix(new double[][] {
                        { -1.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0 },
                        { -0.0, -1.0, -0.0, -0.0, -0.0, -0.0, -0.0 },
                        { -0.0, -0.0, -1.0, -0.0, -0.0, -0.0, -0.0 },
                        { -0.0, -0.0, -0.0, -1.0, -0.0, -0.0, -0.0 },
                        { -0.0, -0.0, -0.0, -0.0, -1.0, -0.0, -0.0 },
                        { -0.0, -0.0, -0.0, -0.0, -0.0, -1.0, -0.0 },
                        { -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -1.0 },
                }),
                new ArrayRealVector(new double[] { -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0 }));


        LinearEqualityConstraint eqc = new LinearEqualityConstraint(
                new double[][] {
                        {
                                1e-06,
                                1e-06,
                                1e-06,
                                1e-06,
                                1e-06,
                                1e-06,
                                1e-06
                        }
                },
                new double[] { 1.0 });

        PointValuePair fpvp = ConvexOptimizer.feasiblePoint(ineqc, eqc);
        assertTrue(fpvp.getSecond() < 0.0);
        double[] ig = fpvp.getFirst();
        BarrierOptimizer barrier = new BarrierOptimizer();
        PointValuePair pvp = barrier.optimize(
                new ObjectiveFunction(q),
                ineqc,
                eqc,
                new InitialGuess(ig)
        );

        double[] RESULT = pvp.getFirst();
        System.out.println("\n\t\t CapitalValue    |        " + Math.round(pvp.getFirst()[0]));
        System.out.println("\t\t GrowthAndIncome |        " + Math.round(pvp.getFirst()[1]));
        System.out.println("\t\t MidCapGrowth    |        " + Math.round(pvp.getFirst()[2]));
        System.out.println("\t\t USGrowth        |        " + Math.round(pvp.getFirst()[3]));
        System.out.println("\t\t Explorer        |        " + Math.round(pvp.getFirst()[4]));
        System.out.println("\t\t WindsorII       |  " + Math.round(pvp.getFirst()[5]));
        System.out.println("\t\t Windsor         |        " + Math.round(pvp.getFirst()[6]));
        assertArrayEquals(TARGET, RESULT, 1e-8);
    }

}
