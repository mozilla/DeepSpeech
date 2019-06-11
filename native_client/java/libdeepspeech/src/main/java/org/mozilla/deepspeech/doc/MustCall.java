package org.mozilla.deepspeech.doc;

import java.lang.annotation.*;

/**
 * Indicates that the annotated method / constructor must invoke a given method
 */
@Documented
@Target({ElementType.METHOD, ElementType.CONSTRUCTOR})
@Retention(RetentionPolicy.SOURCE)
public @interface MustCall {
    /**
     * The name of the method to call
     */
    String value();
}
