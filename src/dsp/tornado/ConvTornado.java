package dsp.tornado;

import dsp.*;
import dsp.cpu.Conv;
import ij.IJ;
import ij.ImageStack;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;
import ijaux.scale.IJLineIteratorIP;
import ijaux.scale.IJLineIteratorStack;

import java.awt.Rectangle;
import java.util.*;

import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * GPU-accelerated convolution backend using TornadoVM.
 *
 * Each convolution type is implemented as a fused TaskGraph: the source image
 * is uploaded once, all passes run on-device (intermediates stay in GPU memory),
 * and only the final result is copied back. Plans are cached by image size and
 * kernel dimensions, so the expensive graph compilation happens only once per
 * geometry — subsequent calls reuse the compiled plan.
 *
 * If any GPU operation fails (driver issue, out of memory, etc.), every method
 * falls back to the CPU implementation in {@link Conv} so the plugin keeps working.
 *
 * @author Dimiter Prodanov IMEC, Sumit Kumar Vohra, Pakhi Awasthi
 *
 * @license LGPL 2.1+
 */
@ConvImplementation(backend = BackendType.GPU)
public class ConvTornado implements IConv, IConv2 {

	public static boolean debug = false;

	public final static int Ox = 0, Oy = 1, Oz = 2;

	//           ===
	//  Plan cache classes — each bundles a compiled TornadoExecutionPlan with
	//  the GPU-visible FloatArray buffers it reads/writes.
	//           ===

	private static final class Conv2DPlan {
		TornadoExecutionPlan plan;
		FloatArray in, k, out;
		int kw, kh;
	}
	private static final HashMap<String, Conv2DPlan> conv2DCache = new HashMap<>();

	/** Structure Tensor gradient block: 1 source → 2 outputs (gradx, grady) */
	private static final class StGradPlan {
		TornadoExecutionPlan plan;
		FloatArray in, kx, kd;
		FloatArray tGx, oGx, tGy, oGy;
	}
	private static final HashMap<String, StGradPlan> stGradCache = new HashMap<>();

	/** Structure Tensor smoothing block: 3 sources → 3 outputs (gx2, gy2, gxy) */
	private static final class StSmoothPlan {
		TornadoExecutionPlan plan;
		FloatArray inA, inB, inC, kx, kd;
		FloatArray tA, oA, tB, oB, tC, oC;
	}
	private static final HashMap<String, StSmoothPlan> stSmoothCache = new HashMap<>();

	/** Single separable convolution: 1 source → 1 output */
	private static final class SepPlan {
		TornadoExecutionPlan plan;
		FloatArray in, kx, kd;
		FloatArray tmp, out;
	}
	private static final HashMap<String, SepPlan> sepCache = new HashMap<>();

	/** Batched 5-output separable convolution (sep3) */
	private static final class Sep3Plan {
		TornadoExecutionPlan plan;
		FloatArray in, kx, kd1, kd2;
		FloatArray tGx, tGy, tXx, tYy, tXy;
		FloatArray oGx, oGy, oXx, oYy, oXy;
	}
	private static final HashMap<String, Sep3Plan> sep3Cache = new HashMap<>();

	/** Semi-separable convolution: 1 source → 1 output (sum of two paths) */
	private static final class SemiSepPlan {
		TornadoExecutionPlan plan;
		FloatArray in, kx, kd;
		FloatArray tA, A, tB, B, out;
	}
	private static final HashMap<String, SemiSepPlan> semiSepCache = new HashMap<>();


	//           ===
	//  GPU kernel functions — annotated with @Parallel for TornadoVM to
	//  parallelize across all pixels. These are the building blocks that
	//  TaskGraphs compose into full convolution pipelines.
	//           ===

	/**
	 * Element-wise addition: out[i] = a[i] + b[i].
	 * Used by convolveSemiSep to sum its two paths.
	 */
	public static void addArr(FloatArray a, FloatArray b, FloatArray out, int n) {
		for (@Parallel int i = 0; i < n; i++) {
			out.set(i, a.get(i) + b.get(i));
		}
	}

	/**
	 * 2D non-separable convolution kernel. Slides the full kw×kh kernel over
	 * every pixel — O(kw·kh) operations per pixel. Clamp-to-edge boundary.
	 * Row-major kernel layout (v outer, u inner).
	 */
	public static void conv2D(FloatArray in, FloatArray out, FloatArray k,
	                           int w, int h, int kw, int kh) {
		final int uc = kw / 2, vc = kh / 2;
		for (@Parallel int y = 0; y < h; y++) {
			for (@Parallel int x = 0; x < w; x++) {
				float sum = 0f;
				int idx = 0;
				for (int v = -vc; v <= vc; v++) {
					int yy = y + v;
					if (yy < 0) yy = 0;
					if (yy >= h) yy = h - 1;
					for (int u = -uc; u <= uc; u++) {
						int xx = x + u;
						if (xx < 0) xx = 0;
						if (xx >= w) xx = w - 1;
						sum += in.get(yy * w + xx) * k.get(idx);
						idx++;
					}
				}
				out.set(y * w + x, sum);
			}
		}
	}

	/**
	 * 1D convolution along rows (X-direction).
	 * @param step  1 for correlation (forward slide), -1 for convolution (backward slide).
	 *              ImageJ's CPU baseline is inconsistent: some filters use correlation,
	 *              others use convolution. The step parameter lets the GPU match each.
	 */
	public static void convX(FloatArray in, FloatArray out, FloatArray k,
	                          int w, int h, int kw, int step) {
		for (@Parallel int y = 0; y < h; y++) {
			for (@Parallel int x = 0; x < w; x++) {
				float sum = 0f;
				int L = k.getSize();
				for (int c = 0; c < L; c++) {
					int idx = x + (c * step) - (kw * step);
					if (idx < 0) idx = 0;
					if (idx >= w) idx = w - 1;
					sum += in.get(y * w + idx) * k.get(c);
				}
				out.set(y * w + x, sum);
			}
		}
	}

	/**
	 * 1D convolution along columns (Y-direction).
	 * @param step  1 for correlation, -1 for convolution (see convX).
	 */
	public static void convY(FloatArray in, FloatArray out, FloatArray k,
	                          int w, int h, int kw, int step) {
		for (@Parallel int y = 0; y < h; y++) {
			for (@Parallel int x = 0; x < w; x++) {
				float sum = 0f;
				int L = k.getSize();
				for (int c = 0; c < L; c++) {
					int idy = y + (c * step) - (kw * step);
					if (idy < 0) idy = 0;
					if (idy >= h) idy = h - 1;
					sum += in.get(idy * w + x) * k.get(c);
				}
				out.set(y * w + x, sum);
			}
		}
	}


	//           ===
	//  GPU convolution methods — each builds/reuses a cached TaskGraph plan
	//           ===

	/**
	 * Batched separable convolution producing five outputs from one source image,
	 * in a single fused GPU execution. Used by filters that need gradients AND
	 * Laplacians simultaneously (Hessian, Curvatures, ALoG, Ridge/Weingarten).
	 *
	 * The five outputs are:
	 *   gradx  = 1st derivative in x, smoothed in y   (d/dx)
	 *   grady  = 1st derivative in y, smoothed in x   (d/dy)
	 *   lap_xx = 2nd derivative in x, smoothed in y   (d2/dx2)
	 *   lap_yy = 2nd derivative in y, smoothed in x   (d2/dy2)
	 *   lap_xy = mixed 2nd derivative                  (d2/dxdy)
	 *
	 * Each output is computed as two separable 1D passes (one per axis),
	 * giving 10 tasks total — all from the same source, all in one TaskGraph.
	 *
	 * Three kernels are used:
	 *   kernx      = Gaussian smoothing kernel
	 *   kern_diff1 = 1st-order derivative kernel (asymmetric)
	 *   kern_diff2 = 2nd-order derivative kernel
	 */
	@Override
	public void convolveSep3(FloatProcessor src, float[] kernx, float[] kern_diff1, float[] kern_diff2,
	                         FloatProcessor gradx, FloatProcessor grady,
	                         FloatProcessor lap_xx, FloatProcessor lap_yy, FloatProcessor lap_xy) {
		final int w = src.getWidth(), h = src.getHeight(), n = w * h;

		try {
			String key = w + "x" + h + ":" + kernx.length + ":" + kern_diff1.length + ":" + kern_diff2.length;
			Sep3Plan p = sep3Cache.get(key);

			if (p == null) {
				p = new Sep3Plan();
				p.in  = new FloatArray(n);
				p.kx  = FloatArray.fromArray(kernx);       // smoothing kernel (uploaded once)
				p.kd1 = FloatArray.fromArray(kern_diff1);   // 1st-derivative kernel (uploaded once)
				p.kd2 = FloatArray.fromArray(kern_diff2);   // 2nd-derivative kernel (uploaded once)

				// For each of the 5 outputs: tXX = scratch (1st pass), oXX = final (2nd pass)
				p.tGx = new FloatArray(n); p.tGy = new FloatArray(n); p.tXx = new FloatArray(n);
				p.tYy = new FloatArray(n); p.tXy = new FloatArray(n);
				p.oGx = new FloatArray(n); p.oGy = new FloatArray(n); p.oXx = new FloatArray(n);
				p.oYy = new FloatArray(n); p.oXy = new FloatArray(n);

				final int kwx = kernx.length / 2, kw1 = kern_diff1.length / 2, kw2 = kern_diff2.length / 2;

				TaskGraph tg = new TaskGraph("sep3_" + key)
					.transferToDevice(DataTransferMode.FIRST_EXECUTION, p.kx, p.kd1, p.kd2)
					.transferToDevice(DataTransferMode.EVERY_EXECUTION, p.in)

					// gradx (d/dx): derivative in X, then smooth in Y
					.task("gx1", ConvTornado::convX, p.in,  p.tGx, p.kd1, w, h, kw1, -1)
					.task("gx2", ConvTornado::convY, p.tGx, p.oGx, p.kx,  w, h, kwx, -1)

					// grady (d/dy): derivative in Y, then smooth in X
					.task("gy1", ConvTornado::convY, p.in,  p.tGy, p.kd1, w, h, kw1, -1)
					.task("gy2", ConvTornado::convX, p.tGy, p.oGy, p.kx,  w, h, kwx, -1)

					// lap_xx (d2/dx2): 2nd derivative in X, then smooth in Y
					.task("xx1", ConvTornado::convX, p.in,  p.tXx, p.kd2, w, h, kw2, -1)
					.task("xx2", ConvTornado::convY, p.tXx, p.oXx, p.kx,  w, h, kwx, -1)

					// lap_yy (d2/dy2): 2nd derivative in Y, then smooth in X
					.task("yy1", ConvTornado::convY, p.in,  p.tYy, p.kd2, w, h, kw2, -1)
					.task("yy2", ConvTornado::convX, p.tYy, p.oYy, p.kx,  w, h, kwx, -1)

					// lap_xy (d2/dxdy): 1st derivative in Y, then 1st derivative in X
					// (mixed partial: differentiate once in each direction, no smoothing)
					.task("xy1", ConvTornado::convY, p.in,  p.tXy, p.kd1, w, h, kw1, -1)
					.task("xy2", ConvTornado::convX, p.tXy, p.oXy, p.kd1, w, h, kw1, -1)

					.transferToHost(DataTransferMode.EVERY_EXECUTION, p.oGx, p.oGy, p.oXx, p.oYy, p.oXy);

				p.plan = new TornadoExecutionPlan(tg.snapshot());
				sep3Cache.put(key, p);
			}

			float[] s = (float[]) src.getPixels();
			for (int i = 0; i < n; i++) p.in.set(i, s[i]);

			p.plan.execute();

			copyBack(p.oGx, gradx);  copyBack(p.oGy, grady);
			copyBack(p.oXx, lap_xx); copyBack(p.oYy, lap_yy); copyBack(p.oXy, lap_xy);

		} catch (Throwable t) {
			IJ.log("convolveSep3 GPU failed, CPU fallback: " + t.getMessage());
			new Conv().convolveSep3(src, kernx, kern_diff1, kern_diff2, gradx, grady, lap_xx, lap_yy, lap_xy);
		}
	}


	/**
	 * Structure Tensor gradient block: from one source, produce two gradient images.
	 *   gradx = X(kern_diff1) → Y(kernx)  (derivative in X, smoothed in Y)
	 *   grady = Y(kern_diff1) → X(kernx)  (derivative in Y, smoothed in X)
	 *
	 * Single fused graph with 4 tasks. Also used by the Gradient filter
	 * (which needs gradients but not the smoothing block).
	 */
	@Override
	public void convolveStructGrad(FloatProcessor src, float[] kernx, float[] kern_diff1,
	                               FloatProcessor gradx, FloatProcessor grady) {
		final int w = src.getWidth(), h = src.getHeight(), n = w * h;

		try {
			String key = "stgrad:" + w + "x" + h + ":" + kernx.length + ":" + kern_diff1.length;
			StGradPlan p = stGradCache.get(key);

			if (p == null) {
				p = new StGradPlan();
				p.in  = new FloatArray(n);
				p.kx  = FloatArray.fromArray(kernx);        // smoothing kernel
				p.kd  = FloatArray.fromArray(kern_diff1);   // derivative kernel
				// tGx/tGy = scratch (1st pass), oGx/oGy = final output (2nd pass)
				p.tGx = new FloatArray(n); p.oGx = new FloatArray(n);
				p.tGy = new FloatArray(n); p.oGy = new FloatArray(n);

				final int kwx = kernx.length / 2, kwd = kern_diff1.length / 2;

				TaskGraph tg = new TaskGraph(key)
					.transferToDevice(DataTransferMode.FIRST_EXECUTION, p.kx, p.kd)
					.transferToDevice(DataTransferMode.EVERY_EXECUTION, p.in)

					// gradx: derivative in X, then smooth in Y
					.task("gx1", ConvTornado::convX, p.in,  p.tGx, p.kd, w, h, kwd, -1)
					.task("gx2", ConvTornado::convY, p.tGx, p.oGx, p.kx, w, h, kwx, -1)

					// grady: derivative in Y, then smooth in X
					.task("gy1", ConvTornado::convY, p.in,  p.tGy, p.kd, w, h, kwd, -1)
					.task("gy2", ConvTornado::convX, p.tGy, p.oGy, p.kx, w, h, kwx, -1)

					.transferToHost(DataTransferMode.EVERY_EXECUTION, p.oGx, p.oGy);

				p.plan = new TornadoExecutionPlan(tg.snapshot());
				stGradCache.put(key, p);
			}

			float[] s = (float[]) src.getPixels();
			for (int i = 0; i < n; i++) p.in.set(i, s[i]);

			p.plan.execute();

			copyBack(p.oGx, gradx);
			copyBack(p.oGy, grady);

		} catch (Throwable t) {
			IJ.log("convolveStructGrad GPU failed, CPU fallback: " + t.getMessage());
			new Conv().convolveStructGrad(src, kernx, kern_diff1, gradx, grady);
		}
	}


	/**
	 * Structure Tensor smoothing block: smooths three gradient-product images.
	 *
	 * Takes three inputs (gx2, gy2, gxy — the squared/cross gradient products
	 * computed by the caller after convolveStructGrad) and Gaussian-smooths each.
	 * This averaging over a local neighbourhood is what produces the actual
	 * structure tensor from raw gradients.
	 *
	 * Single fused graph with 6 tasks (2 passes × 3 inputs). Results written
	 * back in place — each input is overwritten with its smoothed version.
	 */
	@Override
	public void convolveStructSmooth(float[] kernx, float[] kern_diff1,
	                                 FloatProcessor gx2, FloatProcessor gy2, FloatProcessor gxy) {
		final int w = gx2.getWidth(), h = gx2.getHeight(), n = w * h;

		try {
			String key = "stsmooth:" + w + "x" + h + ":" + kernx.length + ":" + kern_diff1.length;
			StSmoothPlan p = stSmoothCache.get(key);

			if (p == null) {
				p = new StSmoothPlan();
				p.inA = new FloatArray(n); p.inB = new FloatArray(n); p.inC = new FloatArray(n);
				p.kx  = FloatArray.fromArray(kernx);
				p.kd  = FloatArray.fromArray(kern_diff1);
				// Scratch (tX) and output (oX) buffers for each of the three inputs
				p.tA = new FloatArray(n); p.oA = new FloatArray(n);
				p.tB = new FloatArray(n); p.oB = new FloatArray(n);
				p.tC = new FloatArray(n); p.oC = new FloatArray(n);

				final int kwx = kernx.length / 2, kwd = kern_diff1.length / 2;

				TaskGraph tg = new TaskGraph(key)
					.transferToDevice(DataTransferMode.FIRST_EXECUTION, p.kx, p.kd)
					.transferToDevice(DataTransferMode.EVERY_EXECUTION, p.inA, p.inB, p.inC)

					// Smooth gx2: X(kern_diff1) → Y(kernx)
					.task("a1", ConvTornado::convX, p.inA, p.tA, p.kd, w, h, kwd, -1)
					.task("a2", ConvTornado::convY, p.tA,  p.oA, p.kx, w, h, kwx, -1)

					// Smooth gy2: Y(kern_diff1) → X(kernx)
					.task("b1", ConvTornado::convY, p.inB, p.tB, p.kd, w, h, kwd, -1)
					.task("b2", ConvTornado::convX, p.tB,  p.oB, p.kx, w, h, kwx, -1)

					// Smooth gxy: Y(kern_diff1) → X(kernx)
					.task("c1", ConvTornado::convY, p.inC, p.tC, p.kd, w, h, kwd, -1)
					.task("c2", ConvTornado::convX, p.tC,  p.oC, p.kx, w, h, kwx, -1)

					.transferToHost(DataTransferMode.EVERY_EXECUTION, p.oA, p.oB, p.oC);

				p.plan = new TornadoExecutionPlan(tg.snapshot());
				stSmoothCache.put(key, p);
			}

			float[] a = (float[]) gx2.getPixels();
			float[] b = (float[]) gy2.getPixels();
			float[] c = (float[]) gxy.getPixels();
			for (int i = 0; i < n; i++) { p.inA.set(i, a[i]); p.inB.set(i, b[i]); p.inC.set(i, c[i]); }

			p.plan.execute();

			copyBack(p.oA, gx2);
			copyBack(p.oB, gy2);
			copyBack(p.oC, gxy);

		} catch (Throwable t) {
			IJ.log("convolveStructSmooth GPU failed, CPU fallback: " + t.getMessage());
			new Conv().convolveStructSmooth(kernx, kern_diff1, gx2, gy2, gxy);
		}
	}


	/**
	 * Single separable convolution: derivative kernel along X, then smoothing
	 * kernel along Y, producing one output in-place. Used by Gaussian Jet.
	 *
	 * Two tasks in one fused TaskGraph; the intermediate stays on-device.
	 */
	@Override
	public void convolveSep(ImageProcessor ip, float[] kernx, float[] kern_diff) {
		final int w = ip.getWidth(), h = ip.getHeight(), n = w * h;

		try {
			String key = "sep:" + w + "x" + h + ":" + kernx.length + ":" + kern_diff.length;
			SepPlan p = sepCache.get(key);

			if (p == null) {
				p = new SepPlan();
				p.in  = new FloatArray(n);
				p.kx  = FloatArray.fromArray(kernx);       // smoothing kernel (uploaded once)
				p.kd  = FloatArray.fromArray(kern_diff);    // derivative kernel (uploaded once)
				p.tmp = new FloatArray(n);                  // scratch: X-pass result, stays on device
				p.out = new FloatArray(n);                  // final output after Y-pass

				final int kwd = kern_diff.length / 2;
				final int kwx = kernx.length / 2;

				TaskGraph tg = new TaskGraph(key)
					.transferToDevice(DataTransferMode.FIRST_EXECUTION, p.kx, p.kd)
					.transferToDevice(DataTransferMode.EVERY_EXECUTION, p.in)

					// Pass 1: derivative kernel along X → scratch
					.task("sx", ConvTornado::convX, p.in,  p.tmp, p.kd, w, h, kwd, 1)
					// Pass 2: smoothing kernel along Y → final output
					.task("sy", ConvTornado::convY, p.tmp, p.out, p.kx, w, h, kwx, 1)

					.transferToHost(DataTransferMode.EVERY_EXECUTION, p.out);

				p.plan = new TornadoExecutionPlan(tg.snapshot());
				sepCache.put(key, p);
			}

			FloatProcessor fp = (ip instanceof FloatProcessor)
					? (FloatProcessor) ip
					: (FloatProcessor) ip.convertToFloat();
			float[] s = (float[]) fp.getPixels();
			for (int i = 0; i < n; i++) p.in.set(i, s[i]);

			p.plan.execute();

			float[] dst = (float[]) fp.getPixels();
			System.arraycopy(p.out.toHeapArray(), 0, dst, 0, n);
			if (ip != fp) ip.setPixels(fp.getPixels());

		} catch (Throwable t) {
			IJ.log("convolveSep GPU failed, CPU fallback: " + t.getMessage());
			new Conv().convolveSep(ip, kernx, kern_diff);
		}
	}


	/**
	 * Semi-separable convolution: sum of two cross-applied separable convolutions.
	 * Used by BoG, Gauss2D, and LoG filters.
	 *
	 * output = Path A + Path B, where:
	 *   Path A = X(kern_diff) → Y(kernx)
	 *   Path B = X(kernx)     → Y(kern_diff)
	 *
	 * 5 tasks fused in one TaskGraph (4 convolution passes + 1 addition).
	 *
	 * ROI: falls back to CPU. The GPU fuses both passes so the intermediate
	 * state outside the ROI differs from CPU's (which only writes inside the
	 * ROI between passes), causing incorrect ROI-edge values. Full-image path
	 * is validated to ~1e-5; ROI uses the CPU implementation.
	 */
	@Override
	public void convolveSemiSep(FloatProcessor ip, float[] kernx, float[] kern_diff) {
		// ROI → CPU fallback (see Javadoc above for the full reasoning)
		java.awt.Rectangle roi = ip.getRoi();
		if (roi != null && (roi.x != 0 || roi.y != 0
				|| roi.width != ip.getWidth() || roi.height != ip.getHeight())) {
			new Conv().convolveSemiSep(ip, kernx, kern_diff);
			return;
		}
//		java.awt.Rectangle roi = ip.getRoi();
//	    boolean fullImage = (roi == null || (roi.x == 0 && roi.y == 0
//	            && roi.width == ip.getWidth() && roi.height == ip.getHeight()));
//
//	    if (!fullImage) {
//	        System.out.println(">>> ROI path firing: roi=" + roi);
//	        convolveSemiSep_roi(ip, kernx, kern_diff, roi);
//	        return;
//	    }
//	    System.out.println(">>> firing for full image");

		final int w = ip.getWidth(), h = ip.getHeight(), n = w * h;

		try {
			String key = "semisep:" + w + "x" + h + ":" + kernx.length + ":" + kern_diff.length;
			SemiSepPlan p = semiSepCache.get(key);

			if (p == null) {
				p = new SemiSepPlan();
				p.in = new FloatArray(n);
				p.kx = FloatArray.fromArray(kernx);       // smoothing kernel (uploaded once)
				p.kd = FloatArray.fromArray(kern_diff);    // derivative kernel (uploaded once)
				// tA/tB = scratch (1st pass), A/B = 2nd pass result, out = A + B
				p.tA = new FloatArray(n); p.A = new FloatArray(n);
				p.tB = new FloatArray(n); p.B = new FloatArray(n);
				p.out = new FloatArray(n);

				final int kwx = kernx.length / 2;
				final int kwd = kern_diff.length / 2;

				TaskGraph tg = new TaskGraph(key)
					.transferToDevice(DataTransferMode.FIRST_EXECUTION, p.kx, p.kd)
					.transferToDevice(DataTransferMode.EVERY_EXECUTION, p.in)

					// Path A: derivative in X, then smooth in Y
					.task("ax", ConvTornado::convX, p.in, p.tA, p.kd, w, h, kwd, 1)
					.task("ay", ConvTornado::convY, p.tA, p.A,  p.kx, w, h, kwx, 1)

					// Path B: smooth in X, then derivative in Y
					.task("bx", ConvTornado::convX, p.in, p.tB, p.kx, w, h, kwx, 1)
					.task("by", ConvTornado::convY, p.tB, p.B,  p.kd, w, h, kwd, 1)

					// Sum: out = A + B
					.task("sum", ConvTornado::addArr, p.A, p.B, p.out, n)

					.transferToHost(DataTransferMode.EVERY_EXECUTION, p.out);

				p.plan = new TornadoExecutionPlan(tg.snapshot());
				semiSepCache.put(key, p);
			}

			float[] s = (float[]) ip.getPixels();
			for (int i = 0; i < n; i++) p.in.set(i, s[i]);

			p.plan.execute();

			System.arraycopy(p.out.toHeapArray(), 0, (float[]) ip.getPixels(), 0, n);

		} catch (Throwable t) {
			IJ.log("convolveSemiSep GPU failed, CPU fallback: " + t.getMessage());
			new Conv().convolveSemiSep(ip, kernx, kern_diff);
		}
	}
	
	/**
	 * ROI-aware GPU semi-separable convolution using boundary extension.
	 * Extracts the ROI + padding with edge-clamped values (StaticCondition
	 * from @dprodanov's dspjava BCFactory pattern), convolves the extended
	 * sub-image on GPU as a standalone full image, then pastes the interior
	 * back into the original image at the ROI position.
	 */
//	private void convolveSemiSep_roi(FloatProcessor ip, float[] kernx, 
//	        float[] kern_diff, java.awt.Rectangle roi) {
//	    int w = ip.getWidth(), h = ip.getHeight();
//	    // How far from the ROI do we need real data?
//	    // Both passes' kernel reach combined, to be safe.
//	    int safeZone = kernx.length + kern_diff.length;
//
//	    // 1. Save original pixels
//	    float[] original = ((float[]) ip.getPixels()).clone();
//
//	    // 2. Fill pixels BEYOND the safe zone with edge-clamped values.
//	    //    Pixels WITHIN the safe zone keep their real values.
//	    float[] px = (float[]) ip.getPixels();
//	    int roiX2 = roi.x + roi.width;
//	    int roiY2 = roi.y + roi.height;
//
//	    for (int y = 0; y < h; y++) {
//	        for (int x = 0; x < w; x++) {
//	            // Inside ROI → keep
//	            if (x >= roi.x && x < roiX2 && y >= roi.y && y < roiY2) continue;
//
//	            // Inside safe zone around ROI → keep real data
//	            if (x >= roi.x - safeZone && x < roiX2 + safeZone &&
//	                y >= roi.y - safeZone && y < roiY2 + safeZone) continue;
//
//	            // Beyond safe zone → clamp to nearest ROI edge pixel
//	            int cx = Math.max(roi.x, Math.min(roiX2 - 1, x));
//	            int cy = Math.max(roi.y, Math.min(roiY2 - 1, y));
//	            px[y * w + x] = px[cy * w + cx];
//	        }
//	    }
//
//	    // 3. Clear ROI so GPU takes full-image path
//	    ip.resetRoi();
//
//	    // 4. GPU convolves the full image
//	    convolveSemiSep(ip, kernx, kern_diff);
//
//	    // 5. Restore: put original pixels back everywhere, then
//	    //    overwrite only the ROI with the convolved result
//	    float[] convolved = ((float[]) ip.getPixels()).clone();
//	    System.arraycopy(original, 0, (float[]) ip.getPixels(), 0, original.length);
//	    for (int y = roi.y; y < roiY2; y++) {
//	        for (int x = roi.x; x < roiX2; x++) {
//	            ((float[]) ip.getPixels())[y * w + x] = convolved[y * w + x];
//	        }
//	    }
//
//	    // 6. Restore the ROI
//	    ip.setRoi(roi);
//	}


	/**
	 * 2D (non-separable) convolution on GPU. Slides the full kw×kh kernel over
	 * every pixel — O(kw·kh) operations per pixel, making this the most
	 * compute-dense convolution type (~24× speedup on Gaussian Derivatives).
	 *
	 * Key difference from separable methods: kernel VALUES change between
	 * filters/scales (different 2D kernels of the same dimensions), so the
	 * kernel is uploaded EVERY_EXECUTION, not FIRST_EXECUTION.
	 *
	 * ROI/mask: falls back to CPU (GPU graph is full-image only).
	 */
	@Override
	public boolean convolveFloat(ImageProcessor ip, float[] kernel, int kw, int kh) {
		final int w = ip.getWidth(), h = ip.getHeight(), n = w * h;
		final java.awt.Rectangle roi = ip.getRoi();

		boolean fullImage = (ip.getMask() == null &&
			roi.x == 0 && roi.y == 0 && roi.width == w && roi.height == h);
		if (!fullImage) {
			return convolveFloatCPU(ip, kernel, kw, kh);
		}

		try {
			String key = "conv2d:" + w + "x" + h + ":" + kw + "x" + kh;
			Conv2DPlan p = conv2DCache.get(key);

			if (p == null) {
				p = new Conv2DPlan();
				p.in  = new FloatArray(n);
				p.k   = new FloatArray(kernel.length);   // 2D kernel (refreshed every call)
				p.out = new FloatArray(n);
				p.kw = kw; p.kh = kh;

				TaskGraph tg = new TaskGraph(key)
					// Both input AND kernel uploaded every call (kernel values differ per filter)
					.transferToDevice(DataTransferMode.EVERY_EXECUTION, p.in, p.k)
					.task("c2d", ConvTornado::conv2D, p.in, p.out, p.k, w, h, kw, kh)
					.transferToHost(DataTransferMode.EVERY_EXECUTION, p.out);

				p.plan = new TornadoExecutionPlan(tg.snapshot());
				conv2DCache.put(key, p);
			}

			float[] s = (float[]) ip.getPixelsCopy();
			for (int i = 0; i < n; i++) p.in.set(i, s[i]);
			for (int i = 0; i < kernel.length; i++) p.k.set(i, kernel[i]);

			p.plan.execute();

			System.arraycopy(p.out.toHeapArray(), 0, (float[]) ip.getPixels(), 0, n);
			return true;

		} catch (Throwable t) {
			IJ.log("convolveFloat GPU failed, CPU fallback: " + t.getMessage());
			return convolveFloatCPU(ip, kernel, kw, kh);
		}
	}


	//           ===
	//  CPU methods — used directly for non-GPU paths (ROI, line-iterator,
	//  3D stacks) and as the reference for GPU fallback via convolveFloatCPU.
	//  These are unchanged from the original Conv implementation.
	//           ===

	/** CPU 2D convolution — used as fallback for convolveFloat when ROI/mask is set. */
	public boolean convolveFloatCPU(ImageProcessor ip, float[] kernel, int kw, int kh) {
		int width = ip.getWidth();
		int height = ip.getHeight();
		Rectangle r = ip.getRoi();
		boolean nonRectRoi = ip.getMask() != null;
		if (nonRectRoi) ip.snapshot();
		int x1 = r.x, y1 = r.y;
		int x2 = x1 + r.width, y2 = y1 + r.height;
		int uc = kw / 2, vc = kh / 2;
		float[] pixels = (float[]) ip.getPixels();
		float[] pixels2 = (float[]) ip.getPixelsCopy();
		double sum;
		int offset, i;
		boolean edgePixel;
		int xedge = width - uc, yedge = height - vc;
		for (int y = y1; y < y2; y++) {
			for (int x = x1; x < x2; x++) {
				sum = 0.0;
				i = 0;
				edgePixel = y < vc || y >= yedge || x < uc || x >= xedge;
				for (int v = -vc; v <= vc; v++) {
					offset = x + (y + v) * width;
					for (int u = -uc; u <= uc; u++) {
						if (edgePixel) {
							if (i >= kernel.length)
								IJ.log("kernel index error: " + i);
							sum += getPixel(x + u, y + v, pixels2, width, height) * kernel[i++];
						} else {
							sum += pixels2[offset + u] * kernel[i++];
						}
					}
				}
				pixels[x + y * width] = (float) (sum);
			}
		}
		if (nonRectRoi) ip.reset(ip.getMask());
		return true;
	}

	@Override
	public void convolveFloat1D(FloatProcessor fp, float[] kernel, int xdir) {
		IJLineIteratorIP<float[]> iter = new IJLineIteratorIP<float[]>(fp, xdir);
		final int width = fp.getWidth();
		final int height = fp.getHeight();
		FloatProcessor ret = new FloatProcessor(width, height);
		int cnt = 0;
		if (debug) { printvector(kernel); System.out.println(); }
		while (iter.hasNext()) {
			final float[] line = iter.next();
			final float[] line2 = lineConvolve(line, kernel, false);
			iter.putLineFloat(ret, line2, cnt, xdir);
			cnt++;
		}
		fp.setPixels(ret.getPixels());
	}

	@Override
	public void convolveFloat1D(ImageStack is, float[] kernel, int xdir) {
		IJLineIteratorStack<float[]> iter = new IJLineIteratorStack<float[]>(is, xdir);
		final int width = is.getWidth();
		final int height = is.getHeight();
		final int depth = is.getSize();
		ImageStack ret = ImageStack.create(width, height, depth, is.getBitDepth());
		int cnt = 0;
		while (iter.hasNext()) {
			final float[] line = iter.next();
			final float[] line2 = lineConvolve(line, kernel, false);
			iter.putLineFloat(ret, line2, cnt, xdir);
			cnt++;
		}
		for (int c = 1; c <= depth; c++) {
			is.setPixels(ret.getPixels(c), c);
		}
	}

	@Override
	public void convolveFloat1D(ImageProcessor ip, float[] kernel, int kw, int kh) {
		int width = ip.getWidth();
		int height = ip.getHeight();
		Rectangle r = ip.getRoi();
		int x1 = r.x, y1 = r.y;
		int x2 = x1 + r.width, y2 = y1 + r.height;
		int uc = kw / 2, vc = kh / 2;
		float[] pixels = (float[]) ip.getPixels();
		float[] pixels2 = (float[]) ip.getPixelsCopy();
		boolean vertical = kw == 1;
		double sum;
		int offset, i;
		boolean edgePixel;
		int xedge = width - uc, yedge = height - vc;
		for (int y = y1; y < y2; y++) {
			for (int x = x1; x < x2; x++) {
				sum = 0.0;
				i = 0;
				if (vertical) {
					edgePixel = y < vc || y >= yedge;
					offset = x + (y - vc) * width;
					for (int v = -vc; v <= vc; v++) {
						if (edgePixel)
							sum += getPixel(x + uc, y + v, pixels2, width, height) * kernel[i++];
						else
							sum += pixels2[offset + uc] * kernel[i++];
						offset += width;
					}
				} else {
					edgePixel = x < uc || x >= xedge;
					offset = x + (y - vc) * width;
					for (int u = -uc; u <= uc; u++) {
						if (edgePixel)
							sum += getPixel(x + u, y + vc, pixels2, width, height) * kernel[i++];
						else
							sum += pixels2[offset + u] * kernel[i++];
					}
				}
				pixels[x + y * width] = (float) (sum);
			}
		}
	}

	@Override
	public void convolveSemiSepIter(FloatProcessor ip, float[] kernx, float[] kern_diff) {
		FloatProcessor ip2 = null;
		FloatProcessor ipx = null;
		final Rectangle roi = ip.getRoi();
		ip2 = (FloatProcessor) ip.duplicate();
		ip2.setRoi(roi);
		ipx = (FloatProcessor) ip.duplicate();
		ipx.setRoi(roi);
		convolveFloat1D(ipx, kern_diff, Ox);
		convolveFloat1D(ipx, kernx, Oy);
		convolveFloat1D(ip2, kernx, Ox);
		convolveFloat1D(ip2, kern_diff, Oy);
		add(ip2, ipx, ip.getRoi());
		ip.setPixels(ip2.getPixels());
	}

	@Override
	public void convolveSepIter(FloatProcessor ip, float[] kernx, float[] kern_diff) {
		convolveFloat1D(ip, kern_diff, Ox);
		convolveFloat1D(ip, kernx, Oy);
	}

	@Override
	public void convolveSemiSep(ImageStack xstack, float[] kernx, float[] kerny, float[] kernz) {
		ImageStack ystack = cloneStack(xstack);
		ImageStack zstack = cloneStack(xstack);
		convolveFloat1D(xstack, kernx, Ox);
		convolveFloat1D(xstack, kerny, Oy);
		convolveFloat1D(xstack, kernz, Oz);
		convolveFloat1D(ystack, kernx, Oy);
		convolveFloat1D(ystack, kerny, Ox);
		convolveFloat1D(ystack, kernz, Oz);
		convolveFloat1D(zstack, kernx, Oz);
		convolveFloat1D(zstack, kerny, Ox);
		convolveFloat1D(zstack, kernz, Oy);
		addToStack(xstack, ystack, zstack);
		ystack = null;
		zstack = null;
	}

	@Override
	public void convolveSep3D(ImageStack xstack, float[] kernx, float[] kern_diffx, float[] kernz) {
		convolveFloat1D(xstack, kern_diffx, Ox);
		convolveFloat1D(xstack, kernx, Oy);
		convolveFloat1D(xstack, kernz, Oz);
	}


	//           ===
	//  Utility methods
	//           ===

	/**
	 * Replaces all pixels outside the ROI with edge-clamped values
	 * (nearest ROI border pixel), so the GPU can convolve the full image
	 * with consistent boundary data around the ROI.
	 */
//	private static void fillOutsideRoiWithEdgeClamped(FloatProcessor ip, java.awt.Rectangle roi) {
//	    int w = ip.getWidth(), h = ip.getHeight();
//	    float[] px = (float[]) ip.getPixels();
//
//	    for (int y = 0; y < h; y++) {
//	        for (int x = 0; x < w; x++) {
//	            // If this pixel is inside the ROI, leave it alone
//	            if (x >= roi.x && x < roi.x + roi.width &&
//	                y >= roi.y && y < roi.y + roi.height) {
//	                continue;
//	            }
//	            // Clamp to nearest ROI edge
//	            int cx = Math.max(roi.x, Math.min(roi.x + roi.width - 1, x));
//	            int cy = Math.max(roi.y, Math.min(roi.y + roi.height - 1, y));
//	            px[y * w + x] = px[cy * w + cx];
//	        }
//	    }
//	}

	/**
	 * Copies only the ROI pixels from the convolved result back into
	 * the destination image, leaving outside-ROI pixels untouched.
	 */
//	private static void copyRoiOnly(FloatProcessor src, FloatProcessor dest, 
//	        java.awt.Rectangle roi) {
//	    int w = src.getWidth();
//	    float[] srcPx = (float[]) src.getPixels();
//	    float[] dstPx = (float[]) dest.getPixels();
//	    for (int y = roi.y; y < roi.y + roi.height; y++) {
//	        for (int x = roi.x; x < roi.x + roi.width; x++) {
//	            dstPx[y * w + x] = srcPx[y * w + x];
//	        }
//	    }
//	}
	
	/** Copies a GPU FloatArray result back into an ImageJ FloatProcessor. */
	private static void copyBack(FloatArray a, FloatProcessor fp) {
		float[] px = (float[]) fp.getPixels();
		System.arraycopy(a.toHeapArray(), 0, px, 0, px.length);
	}

	public void add(ImageProcessor dest, ImageProcessor src, Rectangle r) {
		for (int y = r.y; y < r.y + r.height; y++) {
			for (int x = r.x; x < r.x + r.width; x++) {
				float sum = dest.getf(x, y) + src.getf(x, y);
				dest.setf(x, y, sum);
			}
		}
	}

	private void addToStack(ImageStack dest, ImageStack a, ImageStack b) {
		int bitdepth = dest.getBitDepth();
		if (bitdepth != a.getBitDepth() || a.getBitDepth() != b.getBitDepth()) return;
		final int sz = dest.getSize();
		for (int i = 1; i <= sz; i++) {
			switch (bitdepth) {
				case 8: {
					byte[] p = (byte[]) dest.getPixels(i), pa = (byte[]) a.getPixels(i), pb = (byte[]) b.getPixels(i);
					for (int c = 0; c < p.length; c++) p[c] += pa[c] + pb[c];
					break;
				}
				case 16: {
					short[] p = (short[]) dest.getPixels(i), pa = (short[]) a.getPixels(i), pb = (short[]) b.getPixels(i);
					for (int c = 0; c < p.length; c++) p[c] += pa[c] + pb[c];
					break;
				}
				case 24: {
					int[] p = (int[]) dest.getPixels(i), pa = (int[]) a.getPixels(i), pb = (int[]) b.getPixels(i);
					for (int c = 0; c < p.length; c++) p[c] += pa[c] + pb[c];
					break;
				}
				case 32: {
					float[] p = (float[]) dest.getPixels(i), pa = (float[]) a.getPixels(i), pb = (float[]) b.getPixels(i);
					for (int c = 0; c < p.length; c++) p[c] += pa[c] + pb[c];
					break;
				}
			}
		}
	}

	public static ImageStack cloneStack(ImageStack is) {
		final int width = is.getWidth();
		final int height = is.getHeight();
		Object[] array = is.getImageArray();
		ImageStack ret = ImageStack.create(width, height, array.length, is.getBitDepth());
		Object[] array2 = array.clone();
		int cnt = 1;
		for (Object o : array2) ret.setPixels(o, cnt++);
		ret.update(is.getProcessor(1));
		ret.setRoi(is.getRoi());
		return ret;
	}

	public static void flip(float[] kernel) {
		final int s = kernel.length - 1;
		for (int i = 0; i < kernel.length / 2; i++) {
			final float c = kernel[i];
			kernel[i] = kernel[s - i];
			kernel[s - i] = c;
		}
	}

	public static float[] lineConvolve(float[] arr, float[] kernel, boolean flip) {
		if (flip) flip(kernel);
		float[] y = new float[arr.length];
		int kw = kernel.length / 2;
		for (int i = 0; i < kw; i++) {
			int c = 0;
			for (int k = -kw; k <= kw; k++) {
				int q = i - k;
				if (0 <= q && q < arr.length) { y[i] += arr[q] * kernel[c]; c++; }
				else { y[i] += arr[0] * kernel[c]; c++; }
			}
		}
		for (int i = kw; i < arr.length - kw; i++) {
			int c = 0;
			for (int k = -kw; k <= kw; k++) { y[i] += arr[i - k] * kernel[c]; c++; }
		}
		for (int i = arr.length - kw; i < arr.length; i++) {
			int c = 0;
			for (int k = -kw; k <= kw; k++) {
				int q = i - k;
				if (q < arr.length && 0 <= q) { y[i] += arr[q] * kernel[c]; c++; }
				else { y[i] += arr[arr.length - 1] * kernel[c]; c++; }
			}
		}
		return y;
	}

	private float getPixel(int x, int y, float[] pixels, int width, int height) {
		if (x <= 0) x = 0;
		if (x >= width) x = width - 1;
		if (y <= 0) y = 0;
		if (y >= height) y = height - 1;
		return pixels[x + y * width];
	}

	public static void contrastAdjust(FloatProcessor fpaux, double dr, final double d1) {
		float[] pixels = (float[]) fpaux.getPixels();
		int width = fpaux.getWidth();
		Rectangle rect = fpaux.getRoi();
		for (int i = 0; i < pixels.length; i++) {
			final int x = i % width;
			final int y = i / width;
			if (rect.contains(x, y)) {
				pixels[i] = (float) (pixels[i] * dr + d1);
			}
		}
	}

	public static float[] findMinAndMax(FloatProcessor fp) {
		float[] pixels = (float[]) fp.getPixels();
		int width = fp.getWidth();
		Rectangle rect = fp.getRoi();
		float min = pixels[0];
		float max = min;
		for (int i = 0; i < pixels.length; i++) {
			final int x = i % width;
			final int y = i / width;
			if (rect.contains(x, y)) {
				float value = pixels[i];
				if (!Float.isInfinite(value)) {
					if (value < min) min = value;
					if (value > max) max = value;
				}
			}
		}
		return new float[]{min, max};
	}

	static void printvector(float[] data) {
		for (int i = 0; i < data.length; i++) {
			System.out.print(data[i] + ",");
		}
	}

	public void cleanup() {
	}
}