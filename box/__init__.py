class Box(dict):
    """Minimal stand-in for python-box's Box used in tests."""

    def __getattr__(self, item):
        return self[item]

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, item):
        del self[item]

    def copy(self):  # mimic dict method
        return Box(self)

