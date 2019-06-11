package org.mozilla.deepspeech.doc;


import java.lang.annotation.*;

/**
 * Indicates the DeepSpeech native function that is invoked by the native method that is annotated with this annotation
 */
@Documented
@Target(ElementType.METHOD)
@Retention(RetentionPolicy.SOURCE)
public @interface Calls {

    /**
     * The name of the native function of the DeepSpeech library that is called on invocation of the annotated method
     */
    String value();

}
