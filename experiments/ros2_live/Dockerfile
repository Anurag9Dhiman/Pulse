# Minimal ROS 2 environment for Experiment 17's live integration test.
# ros-base (not -desktop) deliberately: this test only needs rclpy + std_msgs
# pub/sub, not RViz/Gazebo/simulation tooling, and the first attempt at this
# ran the host out of disk pulling a heavier image.
FROM ros:jazzy-ros-base

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY run.sh /run.sh
RUN chmod +x /run.sh

ENTRYPOINT ["/run.sh"]
