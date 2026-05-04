#!/usr/bin/env python3
# Run on the orchestrator; requires dds_data_publisher to forward plans to peers over DDS.
# Set ROBOT_ID on the machine running dds_data_publisher (orchestrator).
"""Publish a MultiRobotGoalPlan (mattbot_dds) for fleet goal batching."""

from __future__ import print_function

import argparse
import sys

import rospy
from mattbot_dds.msg import MultiRobotGoalPlan, RobotGoalEntry


def _parse_goals(s):
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
        "--goals",
        required=True,
        help='Semicolon-separated goals: id,x,y,theta  e.g. "1,1.0,2.0,0;2,3.0,1.0,0"',
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
        goalspec = _parse_goals(args.goals)
    except ValueError as e:
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
