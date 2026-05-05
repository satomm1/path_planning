#!/usr/bin/env python3
# Run on the orchestrator; requires dds_data_publisher to forward plans to peers over DDS.
# Set ROBOT_ID on the machine running dds_data_publisher (orchestrator).
"""Publish a MultiRobotGoalPlan (mattbot_dds) for fleet goal batching."""

from __future__ import print_function

import argparse
import os
import sys

import rospy
from mattbot_dds.msg import MultiRobotGoalPlan, RobotGoalEntry


def _default_goals_file():
    """Path to package config/sample_multi_robot_goals.txt (rospack when available, else next to this script)."""
    try:
        import rospkg

        return os.path.join(rospkg.RosPack().get_path("path_planning"), "config", "sample_multi_robot_goals.txt")
    except Exception:
        here = os.path.dirname(os.path.abspath(__file__))
        return os.path.normpath(os.path.join(here, "..", "config", "sample_multi_robot_goals.txt"))


def _parse_goals_line_format(s):
    """Parse 'id,x,y,th;id2,x2,y2,th2' into list of (int, float, float, float)."""
    out = []
    for part in s.split(";"):
        part = part.strip()
        if not part:
            continue
        bits = [b.strip() for b in part.split(",")]
        if len(bits) != 4:
            raise ValueError("Each goal must be id,x,y,theta (4 fields), got: %r" % (part,))
        rid = int(bits[0])
        x, y, th = float(bits[1]), float(bits[2]), float(bits[3])
        out.append((rid, x, y, th))
    if not out:
        raise ValueError("No goals parsed from %r" % (s,))
    return out


def _parse_goals_file(path):
    """
    Read goals from a text file: one goal per line as id,x,y,theta (commas).
    Lines starting with # and blank lines are ignored. Order matches MultiRobotGoalPlan.goals.
    """
    out = []
    with open(path, "r") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            bits = [b.strip() for b in line.split(",")]
            if len(bits) != 4:
                raise ValueError("%s line %d: expected id,x,y,theta (4 comma-separated fields), got: %r" % (path, lineno, line))
            try:
                rid = int(bits[0])
                x, y, th = float(bits[1]), float(bits[2]), float(bits[3])
            except ValueError as e:
                raise ValueError("%s line %d: invalid number: %s" % (path, lineno, e))
            out.append((rid, x, y, th))
    if not out:
        raise ValueError("No goals in file %r (empty or only comments)" % path)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Publish one MultiRobotGoalPlan to /multi_robot_goal_plan (or --topic)."
    )
    parser.add_argument(
        "--plan_id", default="demo_plan", help="Identifier for this coordinated plan"
    )
    parser.add_argument(
        "--coordinated",
        action="store_true",
        default=True,
        help="Mark as part of multi-robot coordination (default: true)",
    )
    parser.add_argument(
        "--no-coordinated",
        dest="coordinated",
        action="store_false",
        help="Set coordinated=false",
    )
    parser.add_argument(
        "--goals-file",
        "-f",
        metavar="PATH",
        default=_default_goals_file(),
        help="Goals file (default: path_planning/config/sample_multi_robot_goals.txt); one id,x,y,theta per line; # ok",
    )
    parser.add_argument(
        "--goals",
        default=None,
        help='If set, use inline goals instead of --goals-file: id,x,y,theta separated by ";"',
    )
    parser.add_argument(
        "--topic",
        default=None,
        help="Override ROS topic (default: rospy param or /multi_robot_goal_plan)",
    )
    rospy.init_node("publish_multi_robot_goal_plan", anonymous=True)
    argv = rospy.myargv(argv=sys.argv)
    args = parser.parse_args(argv[1:])
    topic = args.topic or rospy.get_param("~topic", "/multi_robot_goal_plan")
    try:
        if args.goals is not None:
            goalspec = _parse_goals_line_format(args.goals)
        else:
            goalspec = _parse_goals_file(args.goals_file)
    except (ValueError, OSError) as e:
        print("Error:", e, file=sys.stderr)
        return 1

    msg = MultiRobotGoalPlan()
    msg.plan_id = args.plan_id
    msg.coordinated = args.coordinated
    msg.goals = []
    for rid, x, y, th in goalspec:
        e = RobotGoalEntry()
        e.robot_id = rid
        e.goal.x = x
        e.goal.y = y
        e.goal.theta = th
        msg.goals.append(e)

    pub = rospy.Publisher(topic, MultiRobotGoalPlan, queue_size=1, latch=True)
    # Wait for at least one subscriber (dds_data_publisher) so the message is not lost
    t0 = rospy.Time.now()
    while pub.get_num_connections() == 0 and not rospy.is_shutdown():
        if (rospy.Time.now() - t0).to_sec() > 5.0:
            rospy.logwarn("No subscribers on %s after 5s; publishing anyway", topic)
            break
        rospy.sleep(0.05)
    pub.publish(msg)
    rospy.loginfo("Published MultiRobotGoalPlan plan_id=%s goals=%d to %s", args.plan_id, len(msg.goals), topic)
    rospy.sleep(0.2)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
