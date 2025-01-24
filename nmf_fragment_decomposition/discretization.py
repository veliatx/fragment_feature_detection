from typing import Literal, Tuple, Union, List

import numpy as np


class MzDiscretize:
    """A class for discretizing mass-to-charge (m/z) values into bins.

    This class provides functionality to create and manage binned m/z values using either
    parts per million (ppm) or dalton (da) bin widths. It supports multiple overlapping
    bin sets.

    Args:
        bin_width (Literal["ppm", "da"]): The type of bin width to use. Defaults to "ppm".
        mz_low (float): The lower m/z boundary. Defaults to 200.0.
        mz_high (float): The upper m/z boundary. Defaults to 2000.0.
        steps (int): Number of overlapping bin sets. Defaults to 2.
        tolerance (float): The bin width in either ppm or da units. Defaults to 60.0/1e6.
    """

    def __init__(
        self,
        bin_width: Literal["ppm", "da"] = "ppm",
        mz_low: float = 200.0,
        mz_high: float = 2000.0,
        steps: int = 2,
        tolerance: float = 60.0 / 1e6,
    ):
        self._bin_width = bin_width
        self._mz_low = mz_low
        self._mz_high = mz_high
        self._steps = steps
        self._tolerance = tolerance
        self._build_bins()

    @property
    def bin_center(self) -> np.ndarray:
        """Get the center values of all bins.

        Returns:
            np.ndarray: Array of bin center values.
        """
        return (self.bin_left_edge + self._bins) / 2

    @property
    def bin_left_edge(self) -> np.ndarray:
        """Get the left edge values of all bins.

        Returns:
            np.ndarray: Array of bin left edge values.
        """
        return np.hstack(
            [
                (self._bins[:, 0] - self._bins[:, 0] * self._tolerance)[
                    ..., np.newaxis
                ],
                self._bins[:, :-1],
            ],
        )

    @property
    def bin_right_edge(self) -> np.ndarray:
        """Get the right edge values of all bins.

        Returns:
            np.ndarray: Array of bin right edge values.
        """
        return self._bins
    
    @property
    def bin_indices(self) -> np.ndarray:
        """Get the indices for all bins across all steps.

        Returns:
            np.ndarray: 2D array of bin indices, shape (steps, bins_per_step).
        """
        m = np.repeat(
            np.arange(self._bins.shape[1])[np.newaxis, ...], 
            self._steps, 
            axis=0
        )
        m += (np.arange(0, self._bins.shape[0]) * self._bins.shape[1])[..., np.newaxis]

        return m

    def indices_to_mz(self, indices: Union[np.ndarray, List[int]], center: bool = False) -> np.ndarray:
        """Convert bin indices to m/z values.

        Args:
            indices (Union[np.ndarray, List[int]]): Bin indices to convert.
            center (bool): If True, return bin center values instead of right edges. Defaults to False.

        Returns:
            np.ndarray: Array of m/z values corresponding to the input indices.
        """
        if not isinstance(indices, np.ndarray):
            indices = np.array(indices)
        
        indices = indices.astype(int)
        
        if center:
            m = self.bin_center
        else:
            m = self.bin_right_edge

        return m.flatten()[indices.flatten()]

    def _build_bins(self) -> None:
        """Build the bin edges based on the specified parameters.

        Creates bins using either constant dalton width or variable ppm width.
        Bins describe right edge of bin, ties go left by default.
        """
        if self._bin_width == "da":
            bins = np.empty(
                (
                    self._steps,
                    np.ceil((self._mz_high - self._mz_low) / self._tolerance).astype(
                        int
                    ),
                )
            )
            for i in range(self._steps):
                bins[i] = np.arange(
                    self._mz_low + (self._tolerance / self._steps * i),
                    self._mz_high + (self._tolerance / self._steps * i),
                    self._tolerance,
                )
        elif self._bin_width == "ppm":
            bins = []
            for i in range(self._steps):
                b = [
                    self._mz_high - (i * self._tolerance * self._mz_high / self._steps)
                ]
                while b[-1] > self._mz_low:
                    b.append(b[-1] - (self._tolerance * b[-1]))
                bins.append(b)
            min_b = min([len(b) for b in bins])
            bins = np.array([b[:min_b][::-1] for b in bins])
        # Bins reflect right edge, filter bins < self._mz_low
        bins = bins[:,bins.min(axis=0) > self._mz_low]
        self._bins = bins

    def discretize(self, m: np.ndarray, center: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Discretize an array of m/z values into bins.

        Args:
            m (np.ndarray): Array of m/z values to discretize.
            center (bool): If True, return bin center values instead of right edges. Defaults to False.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing:
                - Array of bin indices for each input m/z value
                - Array of corresponding bin edge or center values
        """
        row_indices = np.empty((self._steps, m.shape[0]))
        for i in range(self._steps):
            indices = np.searchsorted(self._bins[i], m)
            indices[indices >= self._bins.shape[1]] = self._bins.shape[1] - 1
            # bin_indices[i] = indices + (i * self._bins.shape[1])
            row_indices[i] = indices
        if not center:
            bin = self._bins[
                np.repeat(np.arange(0, self._steps), m.shape[0]),
                row_indices.flatten().astype(int),
            ].reshape(self._steps, -1)

        else:
            bin = self.bin_center[
                np.repeat(np.arange(0, self._steps), m.shape[0]),
                row_indices.flatten().astype(int),
            ].reshape(self._steps, -1)
        
        bin_indices = row_indices + (
            np.arange(0, self._bins.shape[0]) * self._bins.shape[1]
        )[..., np.newaxis]
        return bin_indices, bin

    def discretize_mz_array(
        self,
        m,
        sort_m: bool = True,
        mz_index: int = 3,
    ) -> np.ndarray:
        """Discretize an array containing m/z values and additional information.

        Args:
            m (np.ndarray): 2D array where one column contains m/z values.
            sort_m (bool): If True, sort the array by m/z values. Defaults to True.
            mz_index (int): Column index containing m/z values. Defaults to 3.

        Returns:
            np.ndarray: Discretized array with additional columns for bin indices and edges.
                       Shape is (n_rows * steps, n_columns + 2).
        """
        if sort_m:
            m = m[np.argsort(m[:, mz_index])]

        m = m[(m[:, mz_index] > self._mz_low) & (m[:, mz_index] < self._mz_high)]

        bin_indices, bin_right_edge = self.discretize(m[:, mz_index])

        disc_m = np.empty((m.shape[0] * self._steps, m.shape[1] + 2))
        for i in range(self._steps):
            disc_m[i * m.shape[0] : (i + 1) * m.shape[0]] = np.hstack(
                [
                    m,
                    bin_indices[i][:, np.newaxis],
                    bin_right_edge[i][:, np.newaxis],
                ]
            )

        return disc_m
