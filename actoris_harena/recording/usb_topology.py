"""Which USB hub each rig device hangs off, and why that matters.

A USB hub is a single point of failure for everything plugged into it, and the
rig makes that easy to overlook: cameras, both arms' serial buses and the drive
the dataset is written to are all USB devices, chosen and connected at different
times for unrelated reasons. When such a hub drops -- a brownout, a controller
reset, a knocked cable at the hub end -- every device behind it goes at once, and
the per-device error each one reports says nothing about the shared cause. An
operator reading three "reopen failed" messages sees three broken cameras
rather than one hub.

The worst version is a dataset drive sharing a hub with the sensors recording
into it, because losing that hub does not merely interrupt capture: it removes
the file being written to, mid-episode.

Grouping devices by hub is the whole job here. It is done by the sysfs path,
which spells out the topology: a controller, a bus, then one component per port
hop. ``0000:05:00.4/usb3/3-1/3-1.1/3-1.1.4`` is controller ``0000:05:00.4``,
root port 1, then two hub hops. Devices are grouped by **controller and root
port**, not by the full chain, for two reasons: a nested hub shares the fate of
the hub above it, and a USB3 hub appears as two devices (``3-1`` on the USB2 bus
and ``4-1`` on the USB3 bus) which are one piece of hardware on one power
supply. That grouping was checked against a real failure on this rig, where it
picked out exactly the five devices that died together and none of the ones that
survived.

The parsing is pure and unit-tested; only the lookups touch the filesystem.
"""

from __future__ import annotations

import os
import re
import stat
from pathlib import Path
from typing import NamedTuple

# A sysfs USB device directory: bus number, dash, then dot-separated port hops.
_USB_DIR = re.compile(r"^(\d+)-(\d+(?:\.\d+)*)$")
# A PCI address, e.g. 0000:05:00.4.
_PCI_DIR = re.compile(r"^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f]$")


class UsbLocation(NamedTuple):
    """Where one device sits on the USB tree.

    ``controller`` is the PCI address of its host controller, ``root_port`` the
    port on that controller the whole branch hangs off, and ``chain`` the full
    sysfs port path for reporting. ``hub`` groups devices that share a fate.
    """

    controller: str
    root_port: str
    chain: str

    @property
    def hub(self) -> str:
        """Key identifying the physical hub branch this device depends on."""
        return f"{self.controller}/port{self.root_port}"

    def describe(self) -> str:
        return f"{self.chain} on {self.controller}"


def parse_sysfs_path(realpath: str) -> "UsbLocation | None":
    """Read the USB location out of a resolved sysfs device path. Pure.

    Returns ``None`` for a device that is not behind USB at all (an internal
    camera on PCI, an NVMe drive), which is not a fault: such a device simply
    cannot be taken down by a hub and so never shares one.
    """
    controller = ""
    deepest = ""
    for part in Path(realpath).parts:
        if _PCI_DIR.match(part):
            # The last PCI address before the USB bus is the host controller.
            if not deepest:
                controller = part
        elif _USB_DIR.match(part):
            deepest = part
    if not controller or not deepest:
        return None
    match = _USB_DIR.match(deepest)
    assert match is not None  # guarded by the loop above
    return UsbLocation(
        controller=controller, root_port=match.group(2).split(".")[0], chain=deepest
    )


def device_location(node: "str | Path") -> "UsbLocation | None":
    """Where the device node ``node`` sits on the USB tree, or ``None``.

    Works the same for a camera, a serial port and a block device: every device
    node, character or block, is reachable in sysfs by its major:minor pair, and
    that path spells out the topology. Going through the node rather than a
    ``/dev/*/by-path`` name also means a symlink that has gone stale after a
    re-enumeration resolves to what is actually there now.
    """
    try:
        info = os.stat(node)
        kind = "block" if stat.S_ISBLK(info.st_mode) else "char"
        link = f"/sys/dev/{kind}/{os.major(info.st_rdev)}:{os.minor(info.st_rdev)}"
        return parse_sysfs_path(os.path.realpath(link))
    except OSError:
        return None


def mount_source(directory: "str | Path", mounts: str = "/proc/mounts") -> str:
    """The device backing whichever mount point contains ``directory``.

    The longest matching mount point wins, so a dataset directory on an external
    drive resolves to that drive rather than to the root filesystem it is nested
    under. Returns "" when nothing matches or the table cannot be read.
    """
    try:
        with open(mounts, "r") as f:
            table = f.read()
    except OSError:
        return ""
    return mount_source_from_table(str(Path(directory).resolve()), table)


def mount_source_from_table(directory: str, table: str) -> str:
    """``mount_source`` without the filesystem, for testing. Pure."""
    best_point, best_source = "", ""
    for line in table.splitlines():
        fields = line.split()
        if len(fields) < 2:
            continue
        source, point = fields[0], fields[1].replace("\\040", " ")
        if directory == point or directory.startswith(point.rstrip("/") + "/"):
            if len(point) >= len(best_point):
                best_point, best_source = point, source
    return best_source if best_source.startswith("/dev/") else ""


def directory_location(directory: "str | Path") -> "UsbLocation | None":
    """Where the drive holding ``directory`` sits on the USB tree, or ``None``."""
    source = mount_source(directory)
    return device_location(source) if source else None


def group_by_hub(
    locations: "dict[str, UsbLocation | None]",
) -> "dict[str, list[str]]":
    """``{hub: [labels]}`` for the devices that are on a USB hub at all. Pure.

    Devices with no location are left out rather than lumped together: not being
    on USB is not a shared fate.
    """
    groups: dict[str, list[str]] = {}
    for label, location in locations.items():
        if location is None:
            continue
        groups.setdefault(location.hub, []).append(label)
    return groups
