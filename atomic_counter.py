from threading import Lock


class AtomicCounter:
    def __init__(self, init_value: int = 0):
        self.value = init_value
        self._lock = Lock()

    def increase(self):
        with self._lock:
            self.value += 1

    def decrease(self):
        with self._lock:
            if self.value > 0:
                self.value -= 1

    def increase_and_get(self):
        with self._lock:
            self.value += 1
            return self.value

    def decrease_and_get(self):
        with self._lock:
            if self.value > 0:
                self.value -= 1

            return self.value

    def get(self):
        with self._lock:
            return self.value
