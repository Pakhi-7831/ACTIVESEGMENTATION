package dsp.tornado.tests;

import ij.process.FloatProcessor;

/** Runs every validation test; exits nonzero if any fails. The gating entry point. */
public class RunTests {
    public static void main(String[] args) {
        FloatProcessor src = TestUtil.makeTestImage(256, 256);
        float[] kernx  = TestUtil.gaussian(9);
        float[] kd1    = TestUtil.asymmetric(9);
        float[] kd2    = TestUtil.asymmetric(9);

        boolean all = true;

        System.out.println("\n--- Running ValidateSep3 ---");
        all &= ValidateSep3.run(src, kernx, kd1, kd2);

        System.out.println("\n--- Running ValidateSep ---");
        all &= ValidateSep.run(src, kernx, kd1);

        System.out.println("\n--- Running ValidateSemiSep ---");
        all &= ValidateSemiSep.run(src, kernx, kd1);

        System.out.println("\n--- Running ValidateStruct ---");
        all &= ValidateStruct.run(src, kernx, kd1);

        System.out.println("\n--- Running ValidateConv2D ---");
        all &= ValidateConv2D.run(src, TestUtil.gaussian2D(9), 9, 9);

        System.out.println("\n=== " + (all ? "ALL TESTS PASSED" : "SOME TESTS FAILED") + " ===");
        if (!all) System.exit(1);
    }
}