import logging
import sys
from collections import defaultdict
from typing import List

import numpy as np
from geometry_msgs.msg import Point
from std_msgs.msg import Header
from teaming_msgs.msg import Track


class Hypothesis:
    """hypothesis for single object"""

    def __init__(
        self,
        class_id: int,
        pose: np.ndarray,
        time: float,
        score: float,
        idx: int,
        frame: str,
        label: str = "",
        parent: str = "",
    ) -> None:
        self.n_detections = 0
        self.class_id = class_id
        self.pose = pose
        self.time = time
        self.score = score
        # TODO this will get expensive
        self.poses = [pose]
        self.idx = idx
        self.frame = frame
        self.velocity = np.zeros(3)
        self.history_weight = 0.95
        self.label = label
        self.parent = parent

    def update_pose(self, incoming_pose: np.ndarray) -> np.ndarray:
        return (
            1 - self.history_weight
        ) * incoming_pose + self.history_weight * self.pose

    def update_velocity(self, incoming_pose: np.ndarray) -> np.ndarray:
        return (1 - self.history_weight) * (
            incoming_pose - self.pose
        ) + self.history_weight * self.velocity

    def update(self, time: float, pose: np.ndarray, score: float) -> None:
        self.poses.append(pose)
        self.pose = self.update_pose(pose)
        self.velocity = self.update_velocity(pose)
        self.time = time
        self.score = score
        self.n_detections += 1

    def compute_vel(self, history: int = 25):
        pose_history = np.array(self.poses)[-history:]
        filter = np.ones(5) / 5
        filtered_pose = np.stack(
            [np.convolve(pose_history[:, a], filter) for a in range(3)]
        ).T
        filtered_pose = filtered_pose[4:-4]

        cov = np.cov(filtered_pose.T)
        avg_vel = (filtered_pose[-1:] - filtered_pose[1:]).mean(0)
        return avg_vel, cov

    def is_same(self, incoming_hypothesis, pos_tol: float = 1) -> bool:
        return (
            self.label == incoming_hypothesis.label
            # and self.class_id == incoming_hypothesis.class_id  # use label for open-vocab detection
            and np.linalg.norm(self.pose - incoming_hypothesis.pose) < pos_tol
        )

    def __str__(self):
        print_pose = ", ".join([f"{v:0.2f}" for v in self.pose])
        return f"Track(idx={self.idx}, label={self.label}, id={self.class_id}, pose=({print_pose}, parent={self.parent})"


class HypothesisSet:
    def __init__(self, dist_threshold: float = 1) -> None:
        # TODO make efficient
        self.hypotheses = []
        self.dist_threshold = dist_threshold
        self.depth_threshold = 7.5

    def add_hypothesis(
        self, time: float, class_id: int, score: float, pose: np.ndarray
    ) -> None:
        pass

    def add_detection(
        self,
        time: float,
        class_id: int,
        score: float,
        pose: np.ndarray,
        n_hypothesis: int,
        frame: str,
        label: str,
    ) -> bool:
        # find best fit pose
        added_hypothesis = False
        min_hypothesis_dist = np.inf
        min_idx = -1
        for idx, hypothesis in enumerate(self.hypotheses):
            dist = np.linalg.norm(hypothesis.pose - pose)
            if dist < min_hypothesis_dist:
                min_hypothesis_dist = dist
                min_idx = idx

        if min_hypothesis_dist < self.dist_threshold:
            self.hypotheses[min_idx].update(time=time, pose=pose, score=score)

        else:
            # print(f"min distance: {min_hypothesis_dist:0.2f}. adding new track")
            self.hypotheses.append(
                Hypothesis(
                    class_id=class_id,
                    pose=pose,
                    time=time,
                    score=score,
                    idx=n_hypothesis,
                    frame=frame,
                    label=label,
                )
            )
            added_hypothesis = True
        return added_hypothesis

    def get_status_str(self):
        status = f"hypothesis:\n"
        for hypothesis in self.hypotheses:
            print_pose = ", ".join([f"{v:0.2f}" for v in hypothesis.pose])
            print_avg_vel = ", ".join([f"{v:0.2f}" for v in hypothesis.velocity])

            vel, cov = hypothesis.compute_vel()
            print_vel = ", ".join([f"{v:0.2f}" for v in vel])
            print_cov = ", ".join([f"{v:0.2f}" for v in cov[np.diag_indices(3)]])
            det_cov = np.linalg.det(cov)
            status += (
                f"\tidx: {hypothesis.idx}, class: {hypothesis.class_id}, pose: ({print_pose}), avg vel: ({print_avg_vel})\n"
                f"\t\t velocity: ({print_vel}), cov: ({print_cov}), det: {det_cov:0.2f}, n_dets: {hypothesis.n_detections}"
            )

        status += "\n--"


class Tracker:
    def __init__(self, distance_threshold: float = 1, n_track_thresh: int = 25) -> None:
        class _HypothesisSet(HypothesisSet):  # TODO is there a better way to to this?
            def __init__(self) -> None:
                super().__init__(distance_threshold)

        self.hypotheses = defaultdict(_HypothesisSet)
        self.n_track_thresh = n_track_thresh

        # round about for now
        self.hypothesis_idx = 0

        self.logger = logging.getLogger("Tracker")
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(logging.StreamHandler(sys.stdout))

    def add_detection(
        self,
        time: float,
        class_id: int,
        score: float,
        pose: np.ndarray,
        frame: str,
        label: str = "",
    ) -> None:
        added = self.hypotheses[label].add_detection(
            time=time,
            class_id=class_id,
            score=score,
            pose=pose,
            frame=frame,
            n_hypothesis=self.hypothesis_idx,
            label=label,
        )
        if added:
            self.hypothesis_idx += 1

    def get_tracks(self) -> List[Hypothesis]:
        """Get tracks. A hypothesis is considered a track
        if it as at least `n_trash_thresh` detections associated
        with it

        Returns
        -------
        List[Hypothesis]
            Hypothesis that are now considered tracks.
        """
        tracks = []
        # sets are organized by class. go through each class set
        for hypothesis_set in self.hypotheses.values():
            # check class hypothesis and check if mature
            for hypothesis in hypothesis_set.hypotheses:
                if hypothesis.n_detections > self.n_track_thresh:
                    tracks.append(hypothesis)

        return tracks

    def log_status(self) -> None:
        for hypothesis_set in self.hypotheses.values():
            self.logger.debug(hypothesis_set.get_status_str())


def header_from_track(track: Hypothesis) -> Header:
    header = Header()
    header.frame_id = track.frame
    header.stamp.sec = int(track.time)
    header.stamp.nanosec = int((track.time - int(track.time)) * 1e9)
    return header


def to_track_msg(track: Hypothesis) -> Track:
    track_msg = Track()
    track_msg.header = header_from_track(track)
    track_msg.idx = track.idx
    track_msg.label = track.label
    track_msg.class_id = track.class_id
    track_msg.pose.pose.position = Point(
        x=float(track.pose[0]), y=float(track.pose[1]), z=float(track.pose[2])
    )
    track_msg.pose.pose.orientation.w = 1.0
    return track_msg


def from_track_msg(track_msg: Track, parent: str = "") -> Hypothesis:
    pose = np.array(
        [
            track_msg.pose.pose.position.x,
            track_msg.pose.pose.position.y,
            track_msg.pose.pose.position.z,
        ]
    )
    det_time = track_msg.header.stamp.sec + track_msg.header.stamp.nanosec / 1e9

    return Hypothesis(
        class_id=track_msg.class_id,
        idx=track_msg.idx,
        pose=pose,
        frame=track_msg.header.frame_id,
        time=det_time,
        score=1,
        label=track_msg.label,
        parent=parent,
    )
