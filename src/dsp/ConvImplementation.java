package dsp;

import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;

/**
 * Tags a convolution implementation class with the backend it targets.
 * Read at runtime by ConvFactory to auto-select the right implementation.
 */
@Retention(RetentionPolicy.RUNTIME)
public @interface ConvImplementation {
    BackendType backend();
}