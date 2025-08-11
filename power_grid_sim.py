"""A simple power grid DC power flow simulation.

This script models a small power grid using the DC power flow approximation.
It does not rely on third-party libraries and is meant for educational
purposes only.
"""

from typing import List, Tuple


def build_b_matrix(num_nodes: int, lines: List[Tuple[int, int, float]]) -> List[List[float]]:
    """Construct the susceptance matrix B for the network.

    Args:
        num_nodes: Number of buses in the network.
        lines: List of tuples (i, j, x) where i and j are bus indices and x is
            the reactance of the line connecting them.

    Returns:
        A num_nodes x num_nodes B matrix.
    """
    B = [[0.0 for _ in range(num_nodes)] for _ in range(num_nodes)]
    for i, j, x in lines:
        b = 1 / x
        B[i][i] += b
        B[j][j] += b
        B[i][j] -= b
        B[j][i] -= b
    return B


def solve_linear_system(A: List[List[float]], b: List[float]) -> List[float]:
    """Solve Ax = b for x using Gauss-Jordan elimination.

    This function is intentionally simple and intended for small systems.
    """
    n = len(A)
    # Create an augmented matrix
    M = [row[:] + [b_val] for row, b_val in zip(A, b)]

    for i in range(n):
        # Pivot selection
        pivot_row = max(range(i, n), key=lambda r: abs(M[r][i]))
        M[i], M[pivot_row] = M[pivot_row], M[i]
        pivot = M[i][i]
        if abs(pivot) < 1e-12:
            raise ValueError("Singular matrix")

        # Normalize the pivot row
        for j in range(i, n + 1):
            M[i][j] /= pivot

        # Eliminate other rows
        for r in range(n):
            if r == i:
                continue
            factor = M[r][i]
            for j in range(i, n + 1):
                M[r][j] -= factor * M[i][j]

    return [M[i][-1] for i in range(n)]


def dc_power_flow(
    num_nodes: int,
    lines: List[Tuple[int, int, float]],
    injections: List[float],
) -> Tuple[List[float], List[Tuple[int, int, float]]]:
    """Compute bus voltage angles and line flows for a DC power flow model.

    Args:
        num_nodes: Number of buses.
        lines: List of (i, j, x) tuples describing each transmission line.
        injections: Power injections for each bus (generation positive).

    Returns:
        A tuple containing the list of bus voltage angles and a list of line
        flows as (i, j, flow) tuples.
    """
    if len(injections) != num_nodes:
        raise ValueError("Mismatch between number of buses and injections")

    B = build_b_matrix(num_nodes, lines)

    # Remove the slack bus (bus 0) to solve for other angles
    B_reduced = [row[1:] for row in B[1:]]
    P_reduced = injections[1:]

    angles_unknown = solve_linear_system(B_reduced, P_reduced)
    angles = [0.0] + angles_unknown

    flows = []
    for i, j, x in lines:
        flow = (angles[i] - angles[j]) / x
        flows.append((i, j, flow))

    return angles, flows


def main():
    """Run a sample power flow on a 3-bus system."""
    num_nodes = 3
    lines = [
        (0, 1, 0.1),
        (0, 2, 0.2),
        (1, 2, 0.15),
    ]
    injections = [100.0, -40.0, -60.0]

    angles, flows = dc_power_flow(num_nodes, lines, injections)

    print("Bus voltage angles (radians):")
    for i, angle in enumerate(angles):
        print(f"  Bus {i}: {angle:.4f}")

    print("\nLine flows (MW):")
    for i, j, flow in flows:
        print(f"  {i} -> {j}: {flow:.2f}")


if __name__ == "__main__":
    main()
