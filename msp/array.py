# Multi-dimensional Array Types
class NDArray:
    def __init__(self, kind, dims = []): self.kind, self.dims = kind, dims 
    def __getitem__(self, item): return NDArray(self.kind, self.dims + [item])
