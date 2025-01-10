from enum import Enum

class BodyPart(Enum):
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32

# Mapeamento de partes do corpo detalhadas para categorias simplificadas

simplified_body_parts_mapping = {
    BodyPart.NOSE               : "nose",
    BodyPart.LEFT_EYE_INNER     : "left_eye",
    BodyPart.LEFT_EYE           : "left_eye",
    BodyPart.LEFT_EYE_OUTER     : "left_eye",
    BodyPart.RIGHT_EYE_INNER    : "right_eye",
    BodyPart.RIGHT_EYE          : "right_eye",
    BodyPart.RIGHT_EYE_OUTER    : "right_eye",
    BodyPart.LEFT_EAR           : "left_ear",
    BodyPart.RIGHT_EAR          : "right_ear",
    BodyPart.MOUTH_LEFT         : "mouth",
    BodyPart.MOUTH_RIGHT        : "mouth",
    BodyPart.LEFT_SHOULDER      : "left_shoulder",
    BodyPart.RIGHT_SHOULDER     : "right_shoulder",
    BodyPart.LEFT_ELBOW         : "left_elbow",
    BodyPart.RIGHT_ELBOW        : "right_elbow",
    BodyPart.LEFT_WRIST         : "left_wrist",
    BodyPart.RIGHT_WRIST        : "right_wrist",
    BodyPart.LEFT_PINKY         : "left_hand",
    BodyPart.RIGHT_PINKY        : "right_hand",
    BodyPart.LEFT_INDEX         : "left_hand",
    BodyPart.RIGHT_INDEX        : "right_hand",
    BodyPart.LEFT_THUMB         : "left_hand",
    BodyPart.RIGHT_THUMB        : "right_hand",
    BodyPart.LEFT_HIP           : "left_hand",
    BodyPart.RIGHT_HIP          : "right_hip",
    BodyPart.LEFT_KNEE          : "left_knee",
    BodyPart.RIGHT_KNEE         : "right_knee",
    BodyPart.LEFT_ANKLE         : "left_ankle",
    BodyPart.RIGHT_ANKLE        : "right_ankle",
    BodyPart.LEFT_HEEL          : "left_heel",
    BodyPart.RIGHT_HEEL         : "right_heel",
    BodyPart.LEFT_FOOT_INDEX    : "left_foot",
    BodyPart.RIGHT_FOOT_INDEX   : "right_foot",
}

def simplify_body_parts(pose_landmarks):
    """
    Simplifica os landmarks de pose para categorias genéricas.

    Parameters:
    - pose_landmarks (list): Lista de landmarks detectados.

    Returns:
    - Set com categorias simplificadas de partes do corpo.
    """
    simplified_body_parts = set()
    for id, cx, cy in pose_landmarks:
        body_part = BodyPart(id)
        if body_part in simplified_body_parts_mapping:
            simplified_body_parts.add(simplified_body_parts_mapping[body_part])

    return simplified_body_parts