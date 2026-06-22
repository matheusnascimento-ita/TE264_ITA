# -*- coding: utf-8 -*-
"""Wrapper: importa render() do módulo mamdani na raiz do projeto."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mamdani import render  # noqa: F401
