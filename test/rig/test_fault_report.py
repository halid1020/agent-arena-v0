"""Unit tests for the end-of-session device fault report.

The question under test is who gets blamed. A hub going down surfaces as several
unrelated device errors, and blaming the individual cables sends the operator to
reseat connectors that were never loose; a single device failing repeatedly
really is that device. The reference case is a real failure on this rig where one
hub took two cameras and the dataset drive with it.

Run:  PYTHONPATH=.:src python -m unittest test.unit.test_fault_report
"""

import unittest

from actoris_harena.recording.fault_report import session_fault_report
from actoris_harena.recording.usb_topology import parse_sysfs_path

_PCI = "/sys/devices/pci0000:00/0000:00:08.1"


def _at(chain, controller="0000:05:00.4"):
    bus, hops = chain.split("-")[0], chain.split("-")[1].split(".")
    parts = [f"{bus}-{'.'.join(hops[: i + 1])}" for i in range(len(hops))]
    return f"{_PCI}/{controller}/usb{bus}/" + "/".join(parts)


# The rig as wired when the failure happened: the left arm's devices and the
# drive on one hub, the right arm's on another controller.
LOCATIONS = {
    "camera central": parse_sysfs_path(_at("3-1.2")),
    "camera wrist_camera_left": parse_sysfs_path(_at("3-1.1.4")),
    "follower left": parse_sysfs_path(_at("3-1.4")),
    "dataset drive /mnt/seagate": parse_sysfs_path(_at("4-1.3")),
    "camera wrist_camera_right": parse_sysfs_path(_at("5-1.4", "0000:06:00.4")),
}


class TestSessionFaultReport(unittest.TestCase):
    def test_nothing_failed_says_nothing(self):
        self.assertEqual(session_fault_report({k: 0 for k in LOCATIONS}, LOCATIONS), [])

    def test_several_failures_on_one_hub_blame_the_hub(self):
        lines = session_fault_report(
            {
                "camera central": 1,
                "camera wrist_camera_left": 2,
                "dataset drive /mnt/seagate": 1,
            },
            LOCATIONS,
        )
        text = "\n".join(lines)
        self.assertIn("one USB hub", text)
        self.assertIn("0000:05:00.4/port1", text)
        self.assertNotIn("reseat its connector", text)

    def test_the_hub_report_names_the_devices_still_at_risk(self):
        # The arm did not fail this time, but it is behind the same hub and will
        # go with it next time; the operator needs to know that when choosing a
        # port to move something to.
        lines = session_fault_report(
            {"camera central": 1, "camera wrist_camera_left": 1}, LOCATIONS
        )
        text = "\n".join(lines)
        self.assertIn("also on that hub", text)
        self.assertIn("follower left", text)

    def test_one_device_failing_repeatedly_blames_that_device(self):
        lines = session_fault_report({"camera wrist_camera_right": 4}, LOCATIONS)
        text = "\n".join(lines)
        self.assertIn("4 times", text)
        self.assertIn("reseat its connector", text)
        self.assertNotIn("one USB hub", text)

    def test_a_single_drop_reads_as_a_note_not_an_alarm(self):
        text = "\n".join(session_fault_report({"camera central": 1}, LOCATIONS))
        self.assertIn("once", text)
        self.assertIn("worth a look", text)

    def test_a_lone_failure_reports_where_the_device_sits(self):
        text = "\n".join(session_fault_report({"camera central": 1}, LOCATIONS))
        self.assertIn("3-1.2", text)

    def test_a_hub_fault_and_an_unrelated_one_are_both_reported(self):
        lines = session_fault_report(
            {
                "camera central": 1,
                "camera wrist_camera_left": 1,
                "camera wrist_camera_right": 2,
            },
            LOCATIONS,
        )
        text = "\n".join(lines)
        self.assertIn("one USB hub", text)
        self.assertIn("camera wrist_camera_right dropped off", text)

    def test_a_device_blamed_on_its_hub_is_not_blamed_again_alone(self):
        lines = session_fault_report(
            {"camera central": 1, "camera wrist_camera_left": 1}, LOCATIONS
        )
        self.assertEqual(sum(1 for line in lines if "camera central" in line), 1)

    def test_devices_with_no_known_location_still_get_reported(self):
        lines = session_fault_report({"mystery": 3}, {"mystery": None})
        self.assertEqual(len(lines), 1)
        self.assertIn("mystery", lines[0])


if __name__ == "__main__":
    unittest.main()
