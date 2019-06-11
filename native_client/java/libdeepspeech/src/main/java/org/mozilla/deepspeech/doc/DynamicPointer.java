package org.mozilla.deepspeech.doc;

import java.lang.annotation.*;

/**
 * Indicates that a method returns or stores a pointer into a buffer that is pointing to native dynamic memory that must be freed.
 * {@link #value()} is the name of the method that this pointer must be destroyed with
 */
@Documented
@Target({ElementType.PARAMETER, ElementType.METHOD, ElementType.FIELD})
@Retention(RetentionPolicy.SOURCE)
public @interface DynamicPointer {

    /**
     * The destroyer method that frees the given pointer
     */
    String value();

}
