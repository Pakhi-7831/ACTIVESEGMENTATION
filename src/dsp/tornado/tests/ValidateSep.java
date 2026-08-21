package dsp.tornado.tests;

import dsp.tornado.ConvTornado;
import ij.ImagePlus;
import ij.process.FloatProcessor;

public class ValidateSep {
	
	// validation test for convolveSep function

    public static boolean run(FloatProcessor src, float[] kernx, float[] kern_diff) {
        ConvTornado c = new ConvTornado();
        FloatProcessor cpu = (FloatProcessor) src.duplicate();
        c.convolveFloat1D(cpu, kern_diff, kern_diff.length, 1);
        c.convolveFloat1D(cpu, kernx, 1, kernx.length);

        FloatProcessor gpu = (FloatProcessor) src.duplicate();
        c.convolveSep(gpu, kernx, kern_diff);

        // visualization
        ImagePlus img=new ImagePlus();
        img.setProcessor(cpu);
        img.setTitle("cpu image");
        img.show();
        
        ImagePlus img2=new ImagePlus();
        img2.setProcessor(gpu);
        img2.setTitle("gpu image");
        img2.show();
        
        return TestUtil.check("sep", (float[])cpu.getPixels(), (float[])gpu.getPixels(), src.getWidth());
    }

    public static void main(String[] args) {
        FloatProcessor src = TestUtil.makeTestImage(256, 256);
        if (!run(src, TestUtil.gaussian(9), TestUtil.asymmetric(9))) System.exit(1);
    }
}