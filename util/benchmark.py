#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

def keep_only_digits(s):
    r'''
    local helper to just keep digits
    '''
    fs = ''
    for c in s:
        if c.isdigit():
            fs += c

    return int(fs)
