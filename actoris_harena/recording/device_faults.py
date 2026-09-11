"""Telling a device that has gone away from a device that is merely unhappy.

The rig talks to its motors and cameras over USB, and those two failures look
alike at the call site while calling for opposite responses. A protocol error --
a dropped status packet, a checksum mismatch -- is worth retrying and worth a
stack trace, because it says something about the exchange. A device that has left
the bus is neither: every later call fails the same way, and what the operator
needs to hear is the consequence, not how we found out.

The distinction is drawn from the errno the kernel reports for a device that is
no longer there. Message text is also inspected, because the serial and USB
layers wrap those errnos in their own exception types on the way up and not all
of them preserve ``errno``.

One caution this encodes: once a bus is gone, the port handle itself starts
misbehaving, so a later call can fail with something that names neither the errno
nor the device -- ``port is in use`` was the real example. Such an error must NOT
be classified here; the fact that the bus went is recorded when it is first seen
(``DualDataManager.note_bus_lost``) and consulted afterwards. Guessing from the
downstream symptom would misread a genuine contention error as a missing device.

Pure and unit-tested.
"""

from __future__ import annotations

# Errnos the kernel uses for "this device is not there any more": EIO (the
# transport failed), ENODEV (no such device), ESTALE (the handle outlived it).
_GONE_ERRNOS = frozenset({5, 19, 116})
# Substrings of the messages those errnos travel under once wrapped by pyserial
# or termios, which drop the errno attribute.
_GONE_TEXT = ("no such device", "input/output error")


def bus_gone(exc: BaseException) -> bool:
    """Whether ``exc`` means the device behind it no longer exists. Pure."""
    if isinstance(exc, OSError) and exc.errno in _GONE_ERRNOS:
        return True
    text = str(exc).lower()
    return any(fragment in text for fragment in _GONE_TEXT)
