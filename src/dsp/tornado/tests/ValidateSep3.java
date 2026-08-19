package dsp.tornado.tests;

import dsp.tornado.ConvTornado;
import ij.process.FloatProcessor;

public class ValidateSep3 {

    /** validation test for convolveSep3 function
     * Returns true if all 5 outputs pass. */
    public static boolean run(FloatProcessor src, float[] kernx, float[] kern_diff1, float[] kern_diff2) {
        String[] names = {"gradx", "grady", "lap_xx", "lap_yy", "lap_xy"};
        ConvTornado c = new ConvTornado();

        FloatProcessor cGradx=(FloatProcessor)src.duplicate(), cGrady=(FloatProcessor)src.duplicate(),
                       cLapXx=(FloatProcessor)src.duplicate(), cLapYy=(FloatProcessor)src.duplicate(),
                       cLapXy=(FloatProcessor)src.duplicate();
        c.convolveFloat1D(cGradx, kern_diff1, ConvTornado.Ox);  c.convolveFloat1D(cGradx, kernx, ConvTornado.Oy);
        c.convolveFloat1D(cGrady, kern_diff1, ConvTornado.Oy);  c.convolveFloat1D(cGrady, kernx, ConvTornado.Ox);
        c.convolveFloat1D(cLapXx, kern_diff2, ConvTornado.Ox);  c.convolveFloat1D(cLapXx, kernx, ConvTornado.Oy);
        c.convolveFloat1D(cLapYy, kern_diff2, ConvTornado.Oy);  c.convolveFloat1D(cLapYy, kernx, ConvTornado.Ox);
        c.convolveFloat1D(cLapXy, kern_diff1, ConvTornado.Oy);  c.convolveFloat1D(cLapXy, kern_diff1, ConvTornado.Ox);

        FloatProcessor gGradx=(FloatProcessor)src.duplicate(), gGrady=(FloatProcessor)src.duplicate(),
                       gLapXx=(FloatProcessor)src.duplicate(), gLapYy=(FloatProcessor)src.duplicate(),
                       gLapXy=(FloatProcessor)src.duplicate();
        c.convolveSep3(src, kernx, kern_diff1, kern_diff2, gGradx, gGrady, gLapXx, gLapYy, gLapXy);

        FloatProcessor[] cpu = {cGradx, cGrady, cLapXx, cLapYy, cLapXy};
        FloatProcessor[] gpu = {gGradx, gGrady, gLapXx, gLapYy, gLapXy};
        int w = src.getWidth();
        boolean all = true;
        for (int o=0;o<5;o++)
            all &= TestUtil.check("sep3 "+names[o], (float[])cpu[o].getPixels(), (float[])gpu[o].getPixels(), w);
        return all;
    }

    public static void main(String[] args) {
        FloatProcessor src = TestUtil.makeTestImage(256, 256);
        boolean ok = run(src, TestUtil.gaussian(9), TestUtil.asymmetric(9), TestUtil.asymmetric(9));
        if (!ok) System.exit(1);
    }
}