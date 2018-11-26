#!/bin/bash
../../../bin/lmplz --discount_fallback -o 3 -S 100M --intermediate toy0 --arpa toy0.arpa <<EOF
a a b a
b a a b
EOF
../../../bin/lmplz --discount_fallback -o 3 -S 100M --intermediate toy1 --arpa toy1.arpa <<EOF
a a b b b b b b b
c
EOF
