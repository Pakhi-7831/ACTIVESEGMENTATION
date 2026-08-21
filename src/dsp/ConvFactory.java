package dsp;

import dsp.gpu.ConvGpu;
import dsp.cpu.Conv;
import dsp.tornado.ConvTornado;
import ij.IJ;
import uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider;
import uk.ac.manchester.tornado.api.common.TornadoDevice;
import uk.ac.manchester.tornado.api.TornadoRuntime;
import uk.ac.manchester.tornado.api.enums.TornadoDeviceType;

public class ConvFactory {

    // The three possible engines (unchanged)
    public enum Backend { CPU, GPU, TORNADO }

    // NEW: user's selection mode for the annotation-based auto-selection
    public enum BackendMode { AUTO, FORCE_CPU, FORCE_GPU }

    private static Backend backend = Backend.CPU;
    private static BackendMode mode = BackendMode.AUTO;

    private static ConvGpu gpuInstance = null;
    private static ConvTornado tornadoInstance = null;

    // Candidate implementations for annotation-based resolution.
    // (JCuda ConvGpu is excluded; the modern GPU path is ConvTornado.)
    private static final Class<?>[] CANDIDATES = {
        ConvTornado.class,
        Conv.class
    };

    public static IConv2 createConvApplication() {
        return (IConv2) createConv();
    }

    /** Runtime check: is any GPU-type device available to TornadoVM? */
    public static boolean isGpuAvailable() {
        try {
        	// to trigger the preview feature check immediately
        	Class.forName("uk.ac.manchester.tornado.api.types.arrays.FloatArray");
            TornadoRuntime runtime = TornadoRuntimeProvider.getTornadoRuntime();
            int numBackends = runtime.getNumBackends();
            for (int b = 0; b < numBackends; b++) {
                var runtimeBackend = runtime.getBackend(b);
                int numDevices = runtimeBackend.getNumDevices();
                for (int d = 0; d < numDevices; d++) {
                    TornadoDevice dev = runtimeBackend.getDevice(d);
                    if (dev.getDeviceType() == TornadoDeviceType.GPU) {
                        return true;
                    }
                }
            }
            return false;
        } catch (Throwable t) {
        	if (t.getMessage() != null && t.getMessage().contains("Preview features")) {
                System.out.println("GPU disabled: --enable-preview not set. "
                    + "Add '--enable-preview' to JVM arguments.");
            } else {
                System.out.println("GPU availability check failed: " + t);
            }
            return false;
        }
    }

    /**
     * Decides the target backend type (CPU or GPU) based on the user's mode
     * and, for AUTO, runtime GPU availability.
     */
    private static BackendType decideBackend() {
        switch (mode) {
            case FORCE_CPU:
                return BackendType.CPU;
            case FORCE_GPU:
                if (!isGpuAvailable()) {
                    throw new IllegalStateException(
                        "GPU was explicitly selected but no usable GPU device is available. "
                      + "Switch to Auto or CPU.");
                }
                return BackendType.GPU;
            case AUTO:
            default:
                return isGpuAvailable() ? BackendType.GPU : BackendType.CPU;
        }
    }

    /**
     * Finds the candidate class whose @ConvImplementation matches the wanted
     * backend, reads its annotation via reflection, and instantiates it.
     */
    private static IConv resolveByAnnotation(BackendType wanted) {
        try {
            for (Class<?> c : CANDIDATES) {
                ConvImplementation ann = c.getAnnotation(ConvImplementation.class);
                if (ann != null && ann.backend() == wanted) {
                    // reuse the cached tornado instance if that's what we're building
                    if (c == ConvTornado.class) {
                        if (tornadoInstance == null) tornadoInstance = new ConvTornado();
                        return tornadoInstance;
                    }
                    return (IConv) c.getDeclaredConstructor().newInstance();
                }
            }
            System.out.println("No implementation tagged " + wanted + ", falling back to CPU");
            return new Conv();
        } catch (Exception e) {
            System.out.println("Failed to instantiate " + wanted + " backend, using CPU: " + e);
            return new Conv();
        }
    }

    public static IConv createConv() {
        System.out.println("@@@@@ createConv CALLED, mode=" + mode + " backend=" + backend);

        BackendType wanted = decideBackend();
        // keep the legacy 'backend' field in sync for anything that reads it
        backend = (wanted == BackendType.GPU) ? Backend.TORNADO : Backend.CPU;

        return resolveByAnnotation(wanted);
    }

    public static void setBackend(Backend b) {
        backend = b;
    }

    public static Backend getBackend() {
        return backend;
    }

    // NEW: mode setters for the auto-selection
    public static void setMode(BackendMode m) { mode = m; }
    public static BackendMode getMode() { return mode; }

    public static void cleanup() {
        if (gpuInstance != null) {
            gpuInstance.cleanup();
            gpuInstance = null;
        }
        if (tornadoInstance != null) {
            tornadoInstance.cleanup();
            tornadoInstance = null;
        }
    }

    // Adapter: keep existing setUseGPU callers working, mapped to modes.
    public static void setUseGPU(boolean useGPU) {
        mode = useGPU ? BackendMode.FORCE_GPU : BackendMode.AUTO;
        backend = useGPU ? Backend.TORNADO : Backend.CPU;
    }

    public static boolean isUsingGPU() {
        return backend != Backend.CPU;
    }
}