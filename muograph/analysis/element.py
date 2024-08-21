class Element:
    def __init__(self, density: float, X0: float, name: str):
        self.density = density  # g.cm-3
        self.X0 = X0  # cm
        self.name = name


Al = Element(density=2.699, X0=8.897, name="Aluminum")
Fe = Element(density=7.874, X0=1.57, name="Iron")
Pb = Element(density=11.35, X0=0.5612, name="Lead")
U = Element(density=18.95, X0=0.3166, name="Uranium")
tissue = Element(density=1.127, X0=0.0, name="Tissue")
glass = Element(density=2.23, X0=0.0, name="Glass")
water = Element(density=1.0, X0=36.08, name="Water")
