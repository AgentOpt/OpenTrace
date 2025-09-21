class FIFOBuffer:
    """First-In-First-Out buffer with fixed maximum size.

    Maintains a rolling buffer of items where oldest items are
    automatically removed when the buffer reaches capacity.

    Parameters
    ----------
    size : int
        Maximum number of items to store. If 0, no items are stored.

    Attributes
    ----------
    size : int
        Maximum buffer capacity.
    buffer : list
        Current items in the buffer.

    Methods
    -------
    add(item)
        Add an item to the buffer.
    __iter__()
        Iterate over buffer items.
    __len__()
        Get current number of items.

    Notes
    -----
    Used by optimizers to maintain history of:
    - Recent feedback examples
    - Previous parameter values
    - Optimization trajectories

    When buffer is full, adding a new item removes the oldest
    item (FIFO behavior). This is useful for maintaining a
    sliding window of recent optimization context.

    Examples
    --------
    >>> buffer = FIFOBuffer(size=3)
    >>> for i in range(5):
    ...     buffer.add(i)
    >>> list(buffer)  # Only last 3 items
    [2, 3, 4]
    """
    def __init__(self, size: int):
        self.size = size
        self.buffer = []

    def add(self, item):
        """Add an item to the buffer.

        Parameters
        ----------
        item : Any
            Item to add to the buffer.

        Notes
        -----
        If buffer is at capacity, the oldest item is removed.
        If size is 0, the item is not stored.
        """
        if self.size > 0:
            self.buffer.append(item)
            self.buffer = self.buffer[-self.size:]

    def __iter__(self):
        return iter(self.buffer)

    def __len__(self):
        return len(self.buffer)
