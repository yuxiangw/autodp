import numpy as np
from abc import  ABC, abstractmethod


class Domain:
    def __init__(self, t_min: float, t_max: float, size: int, shifts=0.0) -> None:
        """
        :param float t_min: Coordinate of the node farthest to the left
        :param float t_max: Coordinate of the node farthest to the right
        :param int size: Number of nodes in the domain
        :param float shift: Amount by which this domain has been shifted
        """
        if not isinstance(size, int):
            raise TypeError("`size` must be integer")
        if size % 2 != 0:
            raise ValueError("Must have an even size")
        self._t_min = t_min
        self._size = size
        self._dt = (t_max-t_min)/(size-1)
        self._shifts = shifts

    def __eq__(self, o: "Domain") -> bool:
        return (
            self._t_min == o._t_min and
            self._size == o._size and
            self._shifts == o._shifts and
            self._dt == o._dt
        )

    @abstractmethod
    def create_aligned(t_min: float, t_max: float, dt: float) -> "Domain":
        """
        Create a domain instance that is aligned with the origin.
        The actual domain might be slightly larger than [t_min, t_max]
        but it's guaranteed that the domain is smaller than [t_min-dt, t_max+dt]
        The domain will also be an even size which makes later computing the FFT easier
        :param float t_min: Lower point that will be in the domain
        :param float t_max: Upper point that will be in the domain
        :param float dt: Mesh size
        """
        t_min = np.floor(t_min/dt)*dt
        t_max = np.ceil(t_max/dt)*dt
        size = int(np.round((t_max-t_min)/dt)) + 1
        if size % 2 == 1:
            size += 1
            t_max += dt
        d = Domain(t_min, t_max, size)
        assert np.abs(d.dt() - dt)/dt < 1e-8
        return d

    def shifts(self) -> float:
        """Sum of all shifts that were applied to this domain"""
        return self._shifts

    def shift_right(self, dt: float) -> "Domain":
        """Shift the domain right by `dt`"""
        return Domain(self.t_min()+dt, self.t_max()+dt, len(self), self.shifts() + dt)

    def shift_left(self, dt: float) -> "Domain":
        """Shift the domain left by `dt`"""
        return self.shift_right(-dt)

    def t(self, i: int) -> float:
        return self._t_min + i*self._dt

    def dt(self) -> float:
        return self._dt

    def t_min(self) -> float:
        return self._t_min

    def t_max(self) -> float:
        return self.t(self._size-1)

    def ts(self) -> np.ndarray:
        """Array of all node coordinates in the domain"""
        return np.linspace(self._t_min, self.t_max(), self._size, dtype=np.longdouble, endpoint=True)

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, i: int) -> float:
        return self.t(i)

    def __repr__(self) -> str:
        return f"Domain(t_min={self.t_min()}, t_max={self.t_max()}, size={len(self)}, dt={self.dt()})"



class Discretiser(ABC):
    @abstractmethod
    def discretise(self, prv: PrivacyRandomVariable, domain: Domain) -> DiscretePrivacyRandomVariable:
        pass




class CellCentred(Discretiser):
    def discretise(self, prv: PrivacyRandomVariable, domain: Domain) -> DiscretePrivacyRandomVariable:
        tC = domain.ts()
        tL = tC - domain.dt()/2.0
        tR = tC + domain.dt()/2.0

        # Compute the probability mass cdf(tR) - cdf(tL)
        f = prv.probability(tL, tR)

        mean_d = np.dot(tC, f)
        mean_c = prv.mean()

        mean_shift = mean_c - mean_d

        if not (np.abs(mean_shift) < domain.dt()/2):
            raise RuntimeError("Discrete mean differs from continuous mean significantly.")
        # Shift the domain by the expectation.
        domain_shifted = domain.shift_right(mean_shift)
        return domain_shifted.ts(), f

        # all we need is the shifted domain and its probability mass
        return DiscretePrivacyRandomVariable(f, domain_shifted)



def cdf2disPLD(
             eps_error: float, delta_error: float,
             max_self_compositions: Sequence[int] = None,
             eps_max: Optional[float] = None):
    """
    Privacy Random Variable Accountant for heterogenous composition
    :param prvs: Sequence of `PrivacyRandomVariable` to be composed.
    :type prvs: `Sequence[PrivacyRandomVariable]`
    :param max_self_compositions: Maximum number of compositions of the PRV with itself.
    :type max_self_compositions: Sequence[int]
    :param eps_error: Maximum error allowed in $\varepsilon$. Typically around 0.1
    :param delta_error: Maximum error allowed in $\\delta$. typically around $10^{-3} \times \\delta$
    :param Optional[float] eps_max: Maximum number of valid epsilon. If the true epsilon exceeds this value the
                                    privacy calculation may be off. Setting `eps_max` to `None` automatically computes
                                    a suitable `eps_max` if the PRV supports it.
    """



    if eps_max is not None:
        L = eps_max
        warnings.warn(f"Assuming that true epsilon < {eps_max}. If this is not a valid assumption set `eps_max=None`.")
    else:
        L = compute_safe_domain_size(self.prvs, max_self_compositions, eps_error=eps_error,
                                     delta_error=self.delta_error)

    total_max_self_compositions = sum(max_self_compositions)

    # See Theorem 5.5 in https://arxiv.org/pdf/2106.02848.pdf
    mesh_size = eps_error / np.sqrt(total_max_self_compositions/2*np.log(12/delta_error))
    domain = Domain.create_aligned(-L, L, mesh_size)

    # First do truncation and then discretization
    tprv = PrivacyRandomVariableTruncated(prv, domain.t_min(), domain.t_max())
    dprv = CellCentred().discretise(tprv, domain)
    self.composer = composers.Heterogeneous(dprvs)
