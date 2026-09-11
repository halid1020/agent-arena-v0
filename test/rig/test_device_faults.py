"""Unit tests for classifying a vanished device (actoris_harena.recording.device_faults).

The line drawn here decides what the operator is told when teardown fails: a
device that has gone gets one sentence about the consequence, while a protocol
error keeps its stack trace. Getting it wrong in either direction is bad --
hiding a real bug, or telling someone to power-cycle a rig that is fine.

Run:  PYTHONPATH=.:src python -m unittest test.unit.test_device_faults
"""

import unittest

from actoris_harena.recording.device_faults import bus_gone


class TestBusGone(unittest.TestCase):
    def test_the_errnos_for_a_missing_device(self):
        for errno, name in ((5, "EIO"), (19, "ENODEV"), (116, "ESTALE")):
            self.assertTrue(bus_gone(OSError(errno, name)), name)

    def test_a_wrapped_message_still_counts(self):
        # pyserial and termios re-raise without preserving errno.
        self.assertTrue(bus_gone(Exception("write failed: [Errno 19] No such device")))
        self.assertTrue(bus_gone(Exception("(5, 'Input/output error')")))

    def test_a_protocol_error_is_not_a_missing_device(self):
        # A dropped status packet is worth retrying and worth a stack trace.
        self.assertFalse(
            bus_gone(
                ConnectionError(
                    "Failed to sync read 'Present_Position' on ids=[1] after 1 "
                    "tries. [TxRxResult] There is no status packet!"
                )
            )
        )

    def test_port_contention_is_deliberately_not_classified(self):
        # This is what a wedged handle reports AFTER the bus went, so treating it
        # as proof of a missing device would misread genuine contention with
        # another process. The loss is recorded when first seen instead.
        self.assertFalse(bus_gone(ConnectionError("[TxRxResult] Port is in use!")))

    def test_an_unrelated_error_is_not_a_missing_device(self):
        self.assertFalse(bus_gone(ValueError("bad pose")))
        self.assertFalse(bus_gone(OSError(2, "No such file or directory")))


if __name__ == "__main__":
    unittest.main()
