"""HIMADRI — Hybrid Ice Mapping And Detection using Radar Intelligence.

End-to-end prototype for ISRO BAH 2026 Problem Statement 8:
detection & characterization of subsurface ice in lunar south-polar
doubly-shadowed craters, plus landing-site selection and rover traverse
planning.

The headline idea: a high radar circular-polarisation ratio (CPR) is
ambiguous — rough rocky terrain mimics ice. HIMADRI disambiguates by
fusing CPR with the degree of polarisation (DOP), the m-chi scattering
decomposition, the L/S-band depth signature and optical roughness, and
reports a calibrated per-pixel probability *with uncertainty*.
"""

__version__ = "1.0.0"
