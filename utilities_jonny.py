def calculate_row_column_indices(n, c):
    """
    Function to calculate row and column indices for a given sequential index.

    Parameters:
    - n: Number of plots.
    - c: Number of columns.

    Returns:
    - indices: List of [row, column] indices for each plot.
    """
    return [(i // c, i % c) for i in range(n)]

