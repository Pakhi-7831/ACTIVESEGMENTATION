package dsp.tornado.tests;

import dsp.tornado.ConvTornado;
import ij.ImagePlus;
import ij.process.FloatProcessor;

public class ValidateSemiSep {

    public static boolean run(FloatProcessor src, float[] kernx, float[] kern_diff) {
        boolean ok = runOne(src, kernx, kern_diff, null, "full");
        java.awt.Rectangle roi = new java.awt.Rectangle(
            src.getWidth()/4, src.getHeight()/4, src.getWidth()/2, src.getHeight()/2);
        ok &= runOne(src, kernx, kern_diff, roi, "roi");
        return ok;
    }

    private static boolean runOne(FloatProcessor src, float[] kernx, float[] kern_diff,
                                  java.awt.Rectangle roi, String label) {
        ConvTornado c = new ConvTornado();
        FloatProcessor cpu = (FloatProcessor) src.duplicate();
        if (roi != null) cpu.setRoi(roi);
        {
            FloatProcessor ip2 = (FloatProcessor) cpu.duplicate();
            FloatProcessor ipx = (FloatProcessor) cpu.duplicate();
            if (roi != null) { ip2.setRoi(roi); ipx.setRoi(roi); }
            c.convolveFloat1D(ipx, kern_diff, kern_diff.length, 1);
            c.convolveFloat1D(ipx, kernx, 1, kernx.length);
            c.convolveFloat1D(ip2, kernx, kernx.length, 1);
            c.convolveFloat1D(ip2, kern_diff, 1, kern_diff.length);
            c.add(ip2, ipx, ip2.getRoi());
            cpu.setPixels(ip2.getPixels());
        }
        FloatProcessor gpu = (FloatProcessor) src.duplicate();
        if (roi != null) gpu.setRoi(roi);
        c.convolveSemiSep(gpu, kernx, kern_diff);
        
        // visualization
        ImagePlus img=new ImagePlus();
        img.setProcessor(cpu);
        img.setTitle("cpu image");
        img.show();
        
        ImagePlus img2=new ImagePlus();
        img2.setProcessor(gpu);
        img2.setTitle("gpu image");
        img2.show();

        return TestUtil.check("semiSep["+label+"]", (float[])cpu.getPixels(), (float[])gpu.getPixels(), src.getWidth());
    }

    public static void main(String[] args) {
        FloatProcessor src = TestUtil.makeTestImage(256, 256);     
        if (!run(src, TestUtil.gaussian(9), TestUtil.asymmetric(9))) System.exit(1);
    }
}