package dsp;

import ij.ImageStack;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;

public interface IConv2	 {
    void convolveSep3(FloatProcessor src, float[] kernx, float[] kern_diff1, float[] kern_diff2,FloatProcessor gradx, FloatProcessor grady,FloatProcessor lap_xx, FloatProcessor lap_yy, FloatProcessor lap_xy);
    void convolveStructGrad(FloatProcessor src, float[] kernx, float[] kern_diff1,
            FloatProcessor gradx, FloatProcessor grady);
    void convolveStructSmooth(float[] kernx, float[] kern_diff1,
              FloatProcessor gx2, FloatProcessor gy2, FloatProcessor gxy);
}

