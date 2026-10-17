"""Gemeinsame Fixtures: die Mini-Instanz, an der jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def mini():
    """Mini-Instanz (n = 3, Kern der Länge 3, Nullränder): wahrer Kern kappa* = (0,25, 0,5, 0,25) zu den Versätzen -1, 0, 1, x = (1, 2, 1), also
    A = [[0,5, 0,25, 0], [0,25, 0,5, 0,25], [0, 0,25, 0,5]] und b = A x = (1, 1,5, 1) (Faltung von Hand: 0,5·1 + 0,25·2 = 1; 0,25·1 + 0,5·2 + 0,25·1 = 1,5; 0,25·2 + 0,5·1 = 1).
    Verschobene Funktionen (Randwerte null): S_-1 x = (2, 1, 0), S_0 x = (1, 2, 1), S_1 x = (0, 1, 2); der Kernschritt bei bekanntem x liefert exakt kappa*.
    Trivialpaar (kappa = Delta, x = b / (1 + lam²)): J = lam² |b|² / (1 + lam²) mit |b|² = 4,25, also 0,042079 bei lam = 0,1 und 0,000425 bei lam = 0,01.
    Wahres Paar (Tikhonov x_lam = (A^T A + lam² I)^-1 A^T b = (1,0495, 1,8837, 1,0495) bei lam = 0,1): J = 0,058665 bei lam = 0,1 und 0,000600 bei lam = 0,01: das Trivialpaar hat das kleinere J."""
    kappa = np.array([0.25, 0.5, 0.25])
    x = np.array([1.0, 2.0, 1.0])
    b = np.array([1.0, 1.5, 1.0])
    return kappa, x, b
