package dsp.tornado.tests;

import ij.process.FloatProcessor;
import java.util.Random;

/** Shared helpers for the GPU-vs-CPU validation tests. */
public class TestUtil {

    public static final float TOL = 1e-4f;
    public static final float TOL_2D = 1e-3f;

    /** Compares two pixel arrays; prints result + returns true if within tolerance. */
    public static boolean check(String name, float[] a, float[] b, int w) {
        return check(name, a, b, w, TOL);
    }

    // NEW overload with explicit tolerance
    public static boolean check(String name, float[] a, float[] b, int w, float tol) {
        double max=0, sum=0, meanVal=0; int argmax=-1;
        for (int i=0;i<a.length;i++){
            double d=Math.abs(a[i]-b[i]);
            if(d>max){max=d;argmax=i;}
            sum+=d; meanVal+=Math.abs(a[i]);
        }
        boolean ok = max < tol;
        System.out.println(String.format(
            "%-16s maxDiff=%.6g meanDiff=%.6g meanVal=%.4g worst@(%d,%d)  %s",
            name, max, sum/a.length, meanVal/a.length,
            (argmax>=0?argmax%w:-1), (argmax>=0?argmax/w:-1),
            ok ? "PASS" : "FAIL"));
        return ok;
    }

    /** Fixed-seed reproducible test image. */
    public static FloatProcessor makeTestImage(int w, int h) {
        FloatProcessor fp = new FloatProcessor(w, h);
        float[] px = (float[]) fp.getPixels();
        Random r = new Random(42);
        for (int i=0;i<px.length;i++) px[i] = r.nextFloat()*255f;
        return fp;
    }

    public static float[] gaussian(int len) {
        float[] k = new float[len]; int c=len/2; float s=0;
        for (int i=0;i<len;i++){ k[i]=(float)Math.exp(-(i-c)*(i-c)/4.0); s+=k[i]; }
        for (int i=0;i<len;i++) k[i]/=s;
        return k;
    }

    public static float[] asymmetric(int len) {
        float[] k = new float[len]; int c=len/2;
        for (int i=0;i<len;i++) k[i]=(i-c)*0.1f;
        return k;
    }

    public static float[] gaussian2D(int len) {
        float[] k1=gaussian(len); float[] k=new float[len*len];
        for (int y=0;y<len;y++) for (int x=0;x<len;x++) k[y*len+x]=k1[y]*k1[x];
        return k;
    }
}