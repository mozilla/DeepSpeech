package org.mozilla.deepspeech.doc;

import java.lang.annotation.*;

/**
 * An annotation indicating the c type that a parameter/return actually has.
 * eg. pointers are represented as longs in java
 */
@Documented
@Target({ElementType.PARAMETER, ElementType.METHOD, ElementType.FIELD, ElementType.LOCAL_VARIABLE})
@Retention(RetentionPolicy.SOURCE)
public @interface NativeType {

    /**
     * The c type
     */
    String value();

}
