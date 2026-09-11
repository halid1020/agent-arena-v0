"""Unit tests for USB hub grouping and the preflight topology verdict.

The grouping model is the load-bearing part: devices are grouped by controller
and root port, so a nested hub shares the fate of the hub above it and a USB3
hub's two sysfs faces count as the one piece of hardware they are. The reference
case is a real failure on this rig, where a single hub took a drive, two cameras
and two serial buses with it.

Run:  PYTHONPATH=.:src python -m unittest test.unit.test_usb_topology
"""

import unittest

from actoris_harena.recording.usb_topology import (
    group_by_hub,
    mount_source_from_table,
    parse_sysfs_path,
)

_PCI = "/sys/devices/pci0000:00/0000:00:08.1"


def _at(chain, controller="0000:05:00.4"):
    """A sysfs path for a device at ``chain``, e.g. "3-1.1.4"."""
    bus = chain.split("-")[0]
    hops = chain.split("-")[1].split(".")
    parts = [f"{bus}-{'.'.join(hops[: i + 1])}" for i in range(len(hops))]
    return f"{_PCI}/{controller}/usb{bus}/" + "/".join(parts) + f"/{chain}:1.0"


class TestParseSysfsPath(unittest.TestCase):
    def test_reads_controller_and_root_port(self):
        loc = parse_sysfs_path(_at("3-1.1.4") + "/video4linux/video2")
        self.assertEqual(loc.controller, "0000:05:00.4")
        self.assertEqual(loc.root_port, "1")
        self.assertEqual(loc.chain, "3-1.1.4")

    def test_a_nested_hub_shares_the_fate_of_the_hub_above_it(self):
        deep = parse_sysfs_path(_at("3-1.1.4"))
        shallow = parse_sysfs_path(_at("3-1.2"))
        self.assertEqual(deep.hub, shallow.hub)

    def test_the_two_faces_of_one_usb3_hub_are_one_hub(self):
        # A USB3 hub appears twice, on the USB2 and USB3 buses; it is one piece
        # of hardware on one power supply, and it fails as one.
        usb2 = parse_sysfs_path(_at("3-1.1.4"))
        usb3 = parse_sysfs_path(_at("4-1.3"))
        self.assertEqual(usb2.hub, usb3.hub)

    def test_different_root_ports_are_different_hubs(self):
        self.assertNotEqual(
            parse_sysfs_path(_at("3-1.2")).hub, parse_sysfs_path(_at("3-2")).hub
        )

    def test_different_controllers_are_different_hubs(self):
        self.assertNotEqual(
            parse_sysfs_path(_at("3-1.2")).hub,
            parse_sysfs_path(_at("3-1.2", "0000:06:00.4")).hub,
        )

    def test_a_device_not_behind_usb_has_no_location(self):
        # Not a fault: a PCI or NVMe device cannot be taken down by a hub.
        self.assertIsNone(parse_sysfs_path(f"{_PCI}/0000:05:00.4/nvme/nvme0/nvme0n1"))


class TestGroupByHub(unittest.TestCase):
    def test_unresolved_devices_are_not_a_shared_fate(self):
        groups = group_by_hub(
            {"a": parse_sysfs_path(_at("3-1.2")), "b": None, "c": None}
        )
        self.assertEqual(groups, {"0000:05:00.4/port1": ["a"]})


class TestMountSource(unittest.TestCase):
    TABLE = (
        "/dev/nvme0n1p2 / ext4 rw 0 0\n"
        "/dev/sdb1 /mnt/seagate fuseblk rw 0 0\n"
        "proc /proc proc rw 0 0\n"
    )

    def test_longest_matching_mount_point_wins(self):
        # The dataset directory is nested under the root filesystem too; the
        # answer has to be the drive it actually lives on.
        self.assertEqual(
            mount_source_from_table("/mnt/seagate/so101/cube", self.TABLE),
            "/dev/sdb1",
        )

    def test_a_path_on_the_root_filesystem_resolves_to_it(self):
        self.assertEqual(
            mount_source_from_table("/home/x/ds", self.TABLE), "/dev/nvme0n1p2"
        )

    def test_a_similar_prefix_is_not_a_match(self):
        # /mnt/seagate2 must not match the /mnt/seagate mount point.
        self.assertEqual(
            mount_source_from_table("/mnt/seagate2/ds", self.TABLE), "/dev/nvme0n1p2"
        )

    def test_pseudo_filesystems_are_not_devices(self):
        self.assertEqual(mount_source_from_table("/proc/self", self.TABLE), "")
