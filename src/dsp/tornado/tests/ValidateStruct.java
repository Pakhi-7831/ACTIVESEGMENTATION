package dsp.tornado.tests;

import dsp.tornado.ConvTornado;
import ij.process.FloatProcessor;

public class ValidateStruct {
	
	// validation function for convolveStructGrad and convolveStructSmooth

    public static boolean run(FloatProcessor src, float[] kernx, float[] kern_diff1) {
        ConvTornado c = new ConvTornado();
        int w = src.getWidth();
        boolean ok = true;

        // Block 1: gradients (also covers Gradient filter)
        FloatProcessor oGx=(FloatProcessor)src.duplicate(), oGy=(FloatProcessor)src.duplicate();
        c.convolveFloat1D(oGx, kern_diff1, ConvTornado.Ox); c.convolveFloat1D(oGx, kernx, ConvTornado.Oy);
        c.convolveFloat1D(oGy, kern_diff1, ConvTornado.Oy); c.convolveFloat1D(oGy, kernx, ConvTornado.Ox);

        FloatProcessor nGx=(FloatProcessor)src.duplicate(), nGy=(FloatProcessor)src.duplicate();
        c.convolveStructGrad(src, kernx, kern_diff1, nGx, nGy);

        ok &= TestUtil.check("stGrad gradx", (float[])oGx.getPixels(), (float[])nGx.getPixels(), w);
        ok &= TestUtil.check("stGrad grady", (float[])oGy.getPixels(), (float[])nGy.getPixels(), w);

        // Block 2: smoothing
        FloatProcessor a1=(FloatProcessor)nGx.duplicate(), b1=(FloatProcessor)nGy.duplicate(), c1=(FloatProcessor)nGx.duplicate();
        FloatProcessor a2=(FloatProcessor)nGx.duplicate(), b2=(FloatProcessor)nGy.duplicate(), c2=(FloatProcessor)nGx.duplicate();

        c.convolveFloat1D(a1, kern_diff1, ConvTornado.Ox); c.convolveFloat1D(a1, kernx, ConvTornado.Oy);
        c.convolveFloat1D(b1, kern_diff1, ConvTornado.Oy); c.convolveFloat1D(b1, kernx, ConvTornado.Ox);
        c.convolveFloat1D(c1, kern_diff1, ConvTornado.Oy); c.convolveFloat1D(c1, kernx, ConvTornado.Ox);

        c.convolveStructSmooth(kernx, kern_diff1, a2, b2, c2);

        ok &= TestUtil.check("stSmooth gx2", (float[])a1.getPixels(), (float[])a2.getPixels(), w);
        ok &= TestUtil.check("stSmooth gy2", (float[])b1.getPixels(), (float[])b2.getPixels(), w);
        ok &= TestUtil.check("stSmooth gxy", (float[])c1.getPixels(), (float[])c2.getPixels(), w);
        return ok;
    }

    public static void main(String[] args) {
        FloatProcessor src = TestUtil.makeTestImage(256, 256);
        if (!run(src, TestUtil.gaussian(9), TestUtil.asymmetric(9))) System.exit(1);
    }
}