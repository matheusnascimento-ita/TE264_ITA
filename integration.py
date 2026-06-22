# -*- coding: utf-8 -*-
"""Wrapper: importa render() do módulo integration na raiz do projeto."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from integration import render  # noqa: F401
