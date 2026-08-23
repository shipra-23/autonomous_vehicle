#!/usr/bin/env python3
"""
yahboom_odom_bridge.py

Turns the two usable topics from the Yahboom ROSMASTER driver into inputs that
robot_localization can actually consume.

    /vel_raw       geometry_msgs/Twist   -- no header, no covariance
        --> /wheel/twist   geometry_msgs/TwistWithCovarianceStamped

    /imu/data_raw  sensor_msgs/Imu       -- ALL covariances are zero
        --> /imu/data_cov  sensor_msgs/Imu (covariances filled in)

Why this exists:
  * robot_localization cannot take a bare geometry_msgs/Twist. It needs a
    stamped message with a covariance matrix.
  * Mcnamu_driver.py never sets any covariance on the IMU, so every entry is
    0.0. robot_localization reads a zero variance as "this measurement is
    infinitely certain" and the filter diverges or goes NaN. This node supplies
    realistic values.

This node opens NO serial port. It only reads and republishes topics, so it can
never conflict with the driver.

Usage:
    ros2 run <your_pkg> yahboom_odom_bridge
or standalone:
    python3 yahboom_odom_bridge.py --ros-args -p linear_scale_x:=1.0
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TwistWithCovarianceStamped
from sensor_msgs.msg import Imu


class YahboomOdomBridge(Node):

    def __init__(self):
        super().__init__('yahboom_odom_bridge')

        # ---- parameters -------------------------------------------------
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('imu_frame', 'imu_link')

        # Scale corrections. The STM32's reported velocity is typically a few
        # percent off. Calibrate: drive a measured 2.0 m, compare against the
        # integrated distance, set scale = actual / reported.
        self.declare_parameter('linear_scale_x', 1.0)
        self.declare_parameter('linear_scale_y', 1.0)
        self.declare_parameter('angular_scale_z', 1.0)

        # Set false for a differential-drive or Ackermann base (R2, X1).
        # Leave true only for a mecanum base (X3), where vy is real.
        self.declare_parameter('use_vy', True)

        # Measurement noise as standard deviations; squared into variances.
        self.declare_parameter('vx_std', 0.05)      # m/s
        self.declare_parameter('vy_std', 0.08)      # m/s  (mecanum slips more)
        self.declare_parameter('vyaw_std', 0.08)    # rad/s
        self.declare_parameter('gyro_std', 0.02)    # rad/s
        self.declare_parameter('accel_std', 0.20)   # m/s^2

        def p(name):
            return self.get_parameter(name).value

        self.base_frame = p('base_frame')
        self.imu_frame = p('imu_frame')
        self.scale_x = float(p('linear_scale_x'))
        self.scale_y = float(p('linear_scale_y'))
        self.scale_z = float(p('angular_scale_z'))
        self.use_vy = bool(p('use_vy'))

        # ---- covariance matrices (built once) ---------------------------
        # 6x6 row-major: vx, vy, vz, vroll, vpitch, vyaw
        big = 1e6  # "unobserved" — the filter will ignore these entirely
        self.twist_cov = [0.0] * 36
        self.twist_cov[0] = float(p('vx_std')) ** 2                 # vx
        self.twist_cov[7] = float(p('vy_std')) ** 2 if self.use_vy else big
        self.twist_cov[14] = big                                    # vz
        self.twist_cov[21] = big                                    # vroll
        self.twist_cov[28] = big                                    # vpitch
        self.twist_cov[35] = float(p('vyaw_std')) ** 2              # vyaw

        gyro_var = float(p('gyro_std')) ** 2
        accel_var = float(p('accel_std')) ** 2
        self.gyro_cov = [gyro_var, 0.0, 0.0,
                         0.0, gyro_var, 0.0,
                         0.0, 0.0, gyro_var]
        self.accel_cov = [accel_var, 0.0, 0.0,
                          0.0, accel_var, 0.0,
                          0.0, 0.0, accel_var]

        # REP-145: a leading -1 means "this message carries no orientation".
        # The Yahboom driver publishes gyro + accel only, so we must say so —
        # otherwise robot_localization may try to use an all-zero quaternion.
        self.no_orientation_cov = [-1.0, 0.0, 0.0,
                                   0.0, 0.0, 0.0,
                                   0.0, 0.0, 0.0]

        # ---- pub / sub ---------------------------------------------------
        self.pub_twist = self.create_publisher(
            TwistWithCovarianceStamped, 'wheel/twist', 20)
        self.pub_imu = self.create_publisher(Imu, 'imu/data_cov', 50)

        self.create_subscription(Twist, 'vel_raw', self.on_vel, 20)
        self.create_subscription(Imu, 'imu/data_raw', self.on_imu, 50)

        self.vel_count = 0
        self.imu_count = 0
        self.create_timer(5.0, self.report)

        self.get_logger().info(
            'yahboom_odom_bridge running: '
            '/vel_raw -> /wheel/twist, /imu/data_raw -> /imu/data_cov '
            '(base_frame=%s, use_vy=%s)' % (self.base_frame, self.use_vy))

    # ---------------------------------------------------------------------
    def on_vel(self, msg):
        out = TwistWithCovarianceStamped()
        # /vel_raw has no header, so the best available stamp is arrival time.
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = self.base_frame

        out.twist.twist.linear.x = msg.linear.x * self.scale_x
        out.twist.twist.linear.y = (msg.linear.y * self.scale_y) if self.use_vy else 0.0
        out.twist.twist.linear.z = 0.0
        out.twist.twist.angular.x = 0.0
        out.twist.twist.angular.y = 0.0
        out.twist.twist.angular.z = msg.angular.z * self.scale_z

        out.twist.covariance = self.twist_cov
        self.pub_twist.publish(out)
        self.vel_count += 1

    def on_imu(self, msg):
        if not msg.header.frame_id:
            msg.header.frame_id = self.imu_frame
        msg.orientation_covariance = self.no_orientation_cov
        msg.angular_velocity_covariance = self.gyro_cov
        msg.linear_acceleration_covariance = self.accel_cov
        self.pub_imu.publish(msg)
        self.imu_count += 1

    def report(self):
        if self.vel_count == 0:
            self.get_logger().warn(
                'No /vel_raw messages in the last 5 s — is Mcnamu_driver running?')
        if self.imu_count == 0:
            self.get_logger().warn(
                'No /imu/data_raw messages in the last 5 s.')
        self.vel_count = 0
        self.imu_count = 0


def main(args=None):
    rclpy.init(args=args)
    node = YahboomOdomBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
