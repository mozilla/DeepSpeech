#ifndef LM_MAX_ORDER_H
#define LM_MAX_ORDER_H
/* IF YOUR BUILD SYSTEM PASSES -DKENLM_MAX_ORDER, THEN CHANGE THE BUILD SYSTEM.
 * If not, this is the default maximum order.
 * Having this limit means that State can be
 * (kMaxOrder - 1) * sizeof(float) bytes instead of
 * sizeof(float*) + (kMaxOrder - 1) * sizeof(float) + malloc overhead
 */
#ifndef KENLM_ORDER_MESSAGE
#define KENLM_ORDER_MESSAGE "If your build system supports changing KENLM_MAX_ORDER, change it there and recompile.  In the KenLM tarball or Moses, use e.g. `bjam --max-kenlm-order=6 -a'.  Otherwise, edit lm/max_order.hh."
#endif

#endif // LM_MAX_ORDER_H
