package dsp.tornado.tests;

import dsp.tornado.ConvTornado;
import ij.process.FloatProcessor;

public class ValidateConv2D {

	// validation test for Conv2D
    public static boolean run(FloatProcessor src, float[] kernel, int kw, int kh) {
        ConvTornado c = new ConvTornado();
        FloatProcessor cpu = (FloatProcessor) src.duplicate();
        c.convolveFloatCPU(cpu, kernel, kw, kh);
        FloatProcessor gpu = (FloatProcessor) src.duplicate();
        c.convolveFloat(gpu, kernel, kw, kh);
        return TestUtil.check("conv2D", (float[])cpu.getPixels(), (float[])gpu.getPixels(), src.getWidth(), TestUtil.TOL_2D);    }

    public static void main(String[] args) {
        FloatProcessor src = TestUtil.makeTestImage(256, 256);
        if (!run(src, TestUtil.gaussian2D(9), 9, 9)) System.exit(1);
    }
}